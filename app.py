import csv
import datetime as dt
import io
import json
import os
import random
from pathlib import Path
from urllib.parse import quote

import streamlit as st
from PIL import Image
from dotenv import load_dotenv

from gemini_service import (
    GeminiServiceError,
    KeywordDictItem,
    MiniExampleItem,
    SecurityPlusResponse,
    analyze_and_generate,
    is_rate_limit_error,
    is_timeout_error,
    lookup_keyword,
)

load_dotenv()  # 優先度3: .env（ローカル開発用）を os.environ に読み込む

# 優先度2: Streamlit Secrets の共有デフォルトキーを、.env で未設定の場合のみ os.environ に反映する。
# これは全セッション共通の「アプリ運営者のデフォルトキー」なので process-wide の os.environ に
# 置いても安全（ユーザー個別キーは絶対にここに書き込まない -> st.session_state で管理する）。
if not os.environ.get("GEMINI_API_KEY"):
    try:
        _secret_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        _secret_key = None
    if _secret_key:
        os.environ["GEMINI_API_KEY"] = _secret_key

st.set_page_config(page_title="Security+ Infinite Loop Trainer", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = []
if "labels" not in st.session_state:
    st.session_state.labels = []  # 各階層が何を起点に生成されたかの見出し
if "quick_lookup_cache" not in st.session_state:
    st.session_state.quick_lookup_cache = {}  # 語(小文字) -> KeywordDictItem
if "active_word" not in st.session_state:
    # URLクエリパラメータ(?word=...)と同期する、現在「単語解説」ビューで表示中の単語。
    # None のときは通常の学習フロー（履歴タブ or ホーム画面）を表示する。
    st.session_state.active_word = None
if "obsidian_vault" not in st.session_state:
    st.session_state.obsidian_vault = "Vault"  # ピン留め保存先のObsidian Vault名
if "first_layer_is_derived" not in st.session_state:
    # 第1階層が「設問」由来か「単語シード」由来かを区別し、設問解説タブの見出し・
    # レイアウトを正しく切り替えるためのフラグ。
    st.session_state.first_layer_is_derived = False
if "quickstart_sample" not in st.session_state:
    # 初期画面に表示中のキーワード抜粋（フラットなリスト）。None なら未生成。
    st.session_state.quickstart_sample = None
if "user_api_key" not in st.session_state:
    # 優先度1: ユーザーが個別に入力したAPIキー。セッション単位で保持し、
    # 複数ユーザーが同時アクセスする公開環境でも他人と混ざらないよう
    # 絶対に os.environ には書き込まない。
    st.session_state.user_api_key = ""


def get_effective_api_key() -> str:
    """優先度1(セッション個別キー) > 優先度2/3(os.environに解決済みの共有デフォルトキー)。"""
    return st.session_state.user_api_key.strip() or os.environ.get("GEMINI_API_KEY", "")


# ---------------------------------------------------------------------------
# 単語解説ビューのナビゲーション: st.session_state.active_word と
# URLクエリパラメータ(?word=...)を同期し、リロード・直接アクセス時にも
# 表示中の単語解説を復元できるようにする。
# ---------------------------------------------------------------------------
_qp_word = st.query_params.get("word")
if _qp_word and st.session_state.active_word != _qp_word:
    st.session_state.active_word = _qp_word


def select_word(term: str) -> None:
    """単語解説ビューへ遷移し、URLクエリパラメータに選択中の単語を反映する。"""
    st.session_state.active_word = term
    st.query_params["word"] = term
    st.rerun()


def clear_word_selection() -> None:
    """単語解説ビューを終了し、URLクエリパラメータをクリアして一覧表示に戻る。"""
    st.session_state.active_word = None
    st.query_params.pop("word", None)
    st.rerun()


def _display_gemini_error(exc: GeminiServiceError) -> None:
    """429(レート制限)・タイムアウトは突き放さず、代替手段を案内する親切なメッセージにする。"""
    if is_rate_limit_error(exc):
        if not st.session_state.user_api_key.strip():
            st.error(
                "🚦 現在アクセスが集中しています。数十秒待ってから再度お試しいただくか、"
                "サイドバー下部の「🔑 独自のAPIキーを利用する（任意）」からご自身の無料Gemini APIキーを"
                "設定すると即座に再開できます。"
            )
        else:
            st.error(
                "🚦 ご自身のAPIキーでもレート制限（無料枠の上限など）に達したようです。"
                "数十秒待ってから再度お試しください。"
            )
    elif is_timeout_error(exc):
        st.error(
            f"{exc}\n\n"
            "🔁 もう一度、同じ操作（ボタン）を押して再試行してください。"
            "繰り返しタイムアウトする場合は、サーバー混雑の可能性があります。"
            "数分待つか、サイドバー下部の「🔑 独自のAPIキーを利用する（任意）」から"
            "ご自身のAPIキーをお試しください。"
        )
    else:
        st.error(str(exc))


def _on_gemini_retry(info: dict) -> None:
    """gemini_service側のリトライ発生時に呼ばれ、待機中であることをトースト通知する。"""
    wait_seconds = info.get("wait_seconds")
    wait_str = f"約{wait_seconds:.0f}秒" if wait_seconds is not None else "数秒"
    attempt = info.get("attempt", "?")
    if info.get("is_rate_limit"):
        st.toast(
            f"⏳ APIレート制限の待機中... {wait_str}後に自動再開します（{attempt}回目の試行）",
            icon="⏳",
        )
    elif info.get("is_timeout"):
        st.toast(
            f"⏱️ Gemini APIの応答待ちがタイムアウトしました。{wait_str}後に再試行します（{attempt}回目の試行）",
            icon="⏱️",
        )
    else:
        st.toast(
            f"🔄 Gemini APIへの一時的な接続エラーです。{wait_str}後に自動リトライします（{attempt}回目の試行）",
            icon="🔄",
        )


@st.cache_data
def load_keyword_pool() -> list[str]:
    """data/keywords.json から Security+ 頻出キーワード500語をフラットなリストで読み込む。"""
    path = Path(__file__).parent / "data" / "keywords.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


ALL_QUICKSTART_KEYWORDS = load_keyword_pool()


def sample_quickstart_keywords() -> list[str]:
    """プール(500語)からカテゴリ分けせずに40〜60個をランダム抜粋する。"""
    k = min(random.randint(40, 60), len(ALL_QUICKSTART_KEYWORDS))
    return random.sample(ALL_QUICKSTART_KEYWORDS, k=k)


st.title("🛡️ CompTIA Security+ 循環型語彙・実務トレーニング")

# ---------------------------------------------------------------------------
# 型ガード: key_terms は新形式(KeywordDictItem)・辞書・旧形式(文字列)のいずれでも
# 安全に扱えるよう正規化する（スキーマ更新前に生成された session_state 内の
# 古いデータと混在してもクラッシュしないようにするため）。
# サイドバーより前に定義しておく必要がある（サイドバーのクイック辞書検索から
# derive() / normalize_keyword() を呼び出すため）。
# ---------------------------------------------------------------------------
def normalize_mini_example(me) -> MiniExampleItem:
    if isinstance(me, MiniExampleItem):
        return me
    if isinstance(me, dict):
        return MiniExampleItem(en=str(me.get("en", "")), ja=str(me.get("ja", "")))
    if hasattr(me, "en") and hasattr(me, "ja"):
        return MiniExampleItem(en=str(me.en), ja=str(me.ja))
    return MiniExampleItem(en=str(me), ja="")


def normalize_keyword(kw) -> KeywordDictItem:
    if isinstance(kw, KeywordDictItem):
        return kw
    if isinstance(kw, dict):
        return KeywordDictItem(
            term=str(kw.get("term", kw)),
            meaning=str(kw.get("meaning", "")) or "(旧形式のため情報なし)",
            usage_note=str(kw.get("usage_note", "")) or "(旧形式のため情報なし)",
            collocations=[str(c) for c in kw.get("collocations", [])],
            mini_examples=[normalize_mini_example(me) for me in kw.get("mini_examples", [])],
        )
    term = kw.term if hasattr(kw, "term") else (kw if isinstance(kw, str) else str(kw))
    meaning = getattr(kw, "meaning", "") or "(旧形式のため情報なし)"
    usage_note = getattr(kw, "usage_note", "") or "(旧形式のため情報なし)"
    collocations = [str(c) for c in getattr(kw, "collocations", [])]
    mini_examples = [normalize_mini_example(me) for me in getattr(kw, "mini_examples", [])]
    return KeywordDictItem(
        term=term,
        meaning=meaning,
        usage_note=usage_note,
        collocations=collocations,
        mini_examples=mini_examples,
    )


# ---------------------------------------------------------------------------
# Obsidian連携 & 3段階ピン留め
# ---------------------------------------------------------------------------
OBSIDIAN_NOTE_NAME = "SecurityPlus_Vocab"
# (絵文字, 日本語ラベル, 英語ラベル, タグ)
PIN_LEVELS = [
    ("🔥", "最重要", "High", "P1"),
    ("💡", "要理解", "Medium", "P2"),
    ("📌", "ストック", "Low", "P3"),
]


def format_pin_entry(emoji: str, label_ja: str, term: str, en_line: str, ja_line: str, tag: str) -> str:
    """Obsidianへの保存・コピー用にMarkdown箇条書き1エントリを組み立てる。"""
    date_str = dt.date.today().isoformat()
    lines = [f"- [{emoji} {label_ja}] {term} / {date_str}"]
    if en_line:
        lines.append(f"    - EN: {en_line}")
    if ja_line:
        lines.append(f"    - JA: {ja_line}")
    lines.append(f"    - Tags: #SecurityPlus #{tag}")
    lines.append("")
    return "\n".join(lines)


def build_obsidian_new_uri(vault: str, note: str, content: str) -> str:
    """ネイティブの obsidian://new URIスキームで、既存ノート末尾への追記リンクを組み立てる。"""
    params = {"vault": vault, "file": note, "content": content, "append": "true"}
    query = "&".join(f"{k}={quote(v, safe='')}" for k, v in params.items())
    return f"obsidian://new?{query}"


def render_pin_controls(term: str, en_line: str, ja_line: str, key_prefix: str) -> None:
    """3段階の重要度でObsidianへワンクリック追記保存 + Markdownコピーを提供する。"""
    vault = (st.session_state.obsidian_vault or "Vault").strip() or "Vault"
    with st.popover("📌 ピン留め / Obsidian保存", use_container_width=False, key=f"{key_prefix}_pin_popover"):
        st.caption(f"保存先: **{vault}** vault ／ `{OBSIDIAN_NOTE_NAME}.md`（末尾に追記）")
        entries = {
            label_ja: format_pin_entry(emoji, label_ja, term, en_line, ja_line, tag)
            for emoji, label_ja, _label_en, tag in PIN_LEVELS
        }
        for emoji, label_ja, label_en, tag in PIN_LEVELS:
            uri = build_obsidian_new_uri(vault, OBSIDIAN_NOTE_NAME, entries[label_ja])
            st.link_button(
                f"{emoji} {label_ja} ({label_en}) として保存",
                uri,
                use_container_width=True,
                key=f"{key_prefix}_pin_{tag}",
            )
        st.divider()
        copy_choice = st.radio(
            "📋 コピー用Markdownの重要度",
            [f"{emoji} {label_ja}" for emoji, label_ja, _e, _t in PIN_LEVELS],
            horizontal=True,
            key=f"{key_prefix}_pin_copy_choice",
        )
        chosen_label = copy_choice.split(" ", 1)[1]
        st.code(entries[chosen_label], language="markdown")


def render_keyword_dict(kw: KeywordDictItem, key_prefix: str = "kwdict", show_pin: bool = True) -> None:
    """単語辞書1件分を、popover・単語解説ビュー共通の見た目で表示する。

    show_pin=False は、1つの生成結果内で同じキーワードpopoverが例文の数だけ
    大量に繰り返される場所（実務/ジョーク/恋愛の3タブ分・最大30例文×6語）専用の間引き。
    st.tabs はすべてのタブの中身を毎回まとめて実行するため、ここで各popoverに
    ピン留め(popover+リンクボタン+radio+code)まで積むとウィジェット数が数百に達し、
    生成完了後の初回描画が極端に遅くなるため既定で無効化する。
    """
    st.markdown(f"### {kw.term}")
    st.markdown(f"**一般的な意味・品詞**\n\n{kw.meaning}")
    st.markdown(f"**IT / Security+実務での用法**\n\n{kw.usage_note}")

    if kw.collocations:
        st.markdown("**📌 実務コロケーション**")
        for c in kw.collocations:
            st.markdown(f"- {c}")

    if kw.mini_examples:
        st.markdown("**📝 ミニ実務例文**")
        for n, me in enumerate(kw.mini_examples, 1):
            st.markdown(f"{n}. {me.en}")
            st.caption(me.ja)

    if show_pin:
        first_example = kw.mini_examples[0] if kw.mini_examples else None
        render_pin_controls(
            kw.term,
            en_line=first_example.en if first_example else "",
            ja_line=first_example.ja if first_example else kw.meaning,
            key_prefix=f"{key_prefix}_{kw.term}",
        )


def derive(seed_text: str, key_terms: list[str], prefix: str, index) -> None:
    seed_prompt = f"Target: {seed_text}\nKey Terms: {', '.join(key_terms)}"
    with st.spinner("派生学習コンテンツを生成中..."):
        try:
            new_result = analyze_and_generate(
                seed_prompt,
                mode="derived",
                on_retry=_on_gemini_retry,
                api_key=get_effective_api_key(),
            )
        except GeminiServiceError as exc:
            _display_gemini_error(exc)
            return
    st.session_state.history.append(new_result)
    st.session_state.labels.append(seed_text[:24])
    st.rerun()


def start_from_keyword(keyword: str) -> None:
    """単語1つをシードに、履歴が空でも新規開始・既存履歴があれば派生学習として積む。"""
    st.session_state.first_layer_is_derived = True
    derive(keyword, [keyword], "keyword_seed", keyword)


# ---------------------------------------------------------------------------
# サイドバー：入力エリア
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📥 入力")

    if not get_effective_api_key():
        st.error(
            "⚠️ APIキーが設定されていません。下の「🔑 独自のAPIキーを利用する（任意）」から"
            "ご自身のGemini APIキーを入力してください。"
        )

    uploaded_file = st.file_uploader("問題画像をアップロード", type=["png", "jpg", "jpeg"])
    text_input = st.text_area("または問題文/単語を入力")

    if st.button("🚀 解析・例文生成", type="primary"):
        if not uploaded_file and not text_input:
            st.warning("画像またはテキストを入力してください。")
            st.stop()
        with st.spinner("Geminiが解析＆30例文を生成中..."):
            try:
                if uploaded_file:
                    img = Image.open(uploaded_file)
                    result = analyze_and_generate(
                        img,
                        is_image=True,
                        on_retry=_on_gemini_retry,
                        api_key=get_effective_api_key(),
                    )
                    label = "元の設問（画像）"
                else:
                    result = analyze_and_generate(
                        text_input,
                        is_image=False,
                        on_retry=_on_gemini_retry,
                        api_key=get_effective_api_key(),
                    )
                    label = text_input.strip()[:24] or "元の設問"
            except GeminiServiceError as exc:
                _display_gemini_error(exc)
                st.stop()
        st.session_state.history = [result]
        st.session_state.labels = [label]
        st.session_state.first_layer_is_derived = False
        st.rerun()

    if st.session_state.history:
        st.divider()
        if len(st.session_state.history) > 1:
            if st.button("⬅️ 前の階層に戻る"):
                st.session_state.history.pop()
                st.session_state.labels.pop()
                st.rerun()
        if st.button("🏠 最初からやり直す"):
            st.session_state.history = []
            st.session_state.labels = []
            st.session_state.first_layer_is_derived = False
            clear_word_selection()

    st.divider()
    st.subheader("🔍 クイック単語・熟語検索")
    st.caption("検索結果はメイン画面の「単語解説」ビューに表示され、URLで復元・共有できます。")
    lookup_term = st.text_input("単語・熟語を入力", key="quick_lookup_input")
    if st.button("調べる", key="quick_lookup_button"):
        term = lookup_term.strip()
        if not term:
            st.warning("単語・熟語を入力してください。")
        else:
            cache_key = term.lower()
            if cache_key not in st.session_state.quick_lookup_cache:
                with st.spinner(f'"{term}" を検索中...'):
                    try:
                        looked_up = lookup_keyword(
                            term, on_retry=_on_gemini_retry, api_key=get_effective_api_key()
                        )
                    except GeminiServiceError as exc:
                        _display_gemini_error(exc)
                        looked_up = None
                if looked_up is not None:
                    st.session_state.quick_lookup_cache[cache_key] = looked_up
            if cache_key in st.session_state.quick_lookup_cache:
                looked_up_kw = normalize_keyword(st.session_state.quick_lookup_cache[cache_key])
                select_word(looked_up_kw.term)

    st.divider()
    with st.expander("🔑 独自のAPIキーを利用する（任意）"):
        st.caption(
            "デフォルトでそのままご利用いただけます。アクセス集中時や無制限に使いたい場合は、"
            "ご自身の無料Gemini APIキーを設定してください。"
        )
        st.text_input(
            "あなたのGemini APIキー",
            type="password",
            key="user_api_key",
            placeholder="AIza...",
        )
        st.markdown("[🔗 Google AI Studio で無料のAPIキーを取得](https://aistudio.google.com/apikey)")

    with st.expander("📓 Obsidian連携設定"):
        st.caption(
            f"各例文・単語解説の「📌 ピン留め / Obsidian保存」ボタンから、指定Vaultの "
            f"`{OBSIDIAN_NOTE_NAME}.md` にMarkdown形式で追記保存できます"
            "（お使いの端末にObsidianデスクトップアプリが必要です）。"
        )
        st.text_input("Vault名", key="obsidian_vault", placeholder="Vault")

# ---------------------------------------------------------------------------
# エクスポート用ヘルパー
# ---------------------------------------------------------------------------
def format_keywords(key_terms) -> str:
    return ", ".join(normalize_keyword(kw).term for kw in key_terms)


def build_csv(result: SecurityPlusResponse) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["category", "no", "term_or_en", "meaning_or_jp", "note_or_keywords"])

    writer.writerow(["和訳", "", result.translation, "", ""])
    writer.writerow(["正解", "", result.correct_answer, "", ""])

    for i, v in enumerate(result.vocab_list, 1):
        writer.writerow(["重要語彙", i, v.term, v.meaning, v.exam_point])
    for i, t in enumerate(result.tech_terms, 1):
        writer.writerow(["IT専門用語", i, t.term, t.concept, t.exam_focus])
    for i, ex in enumerate(result.business_examples, 1):
        writer.writerow(["実務例文", i, ex.en, ex.jp, format_keywords(ex.key_terms)])
    for i, ex in enumerate(result.humor_examples, 1):
        writer.writerow(["ITジョーク", i, ex.en, ex.jp, format_keywords(ex.key_terms)])
    for i, ex in enumerate(result.romance_examples, 1):
        writer.writerow(["恋愛ウィット", i, ex.en, ex.jp, format_keywords(ex.key_terms)])

    for i, ex in enumerate(
        result.business_examples + result.humor_examples + result.romance_examples, 1
    ):
        for kw in ex.key_terms:
            kw = normalize_keyword(kw)
            writer.writerow(["クイック辞書", f"{i}:{kw.term}", kw.term, kw.meaning, kw.usage_note])
            if kw.collocations:
                writer.writerow(
                    ["コロケーション", f"{i}:{kw.term}", kw.term, "; ".join(kw.collocations), ""]
                )
            for n, me in enumerate(kw.mini_examples, 1):
                writer.writerow(["ミニ実務例文", f"{i}:{kw.term}:{n}", me.en, me.ja, ""])

    return buf.getvalue().encode("utf-8-sig")


def build_txt(result: SecurityPlusResponse, label: str) -> bytes:
    lines: list[str] = []
    lines.append(f"■ 学習テーマ: {label}")
    lines.append("=" * 60)
    lines.append("【和訳】")
    lines.append(result.translation)
    lines.append("")
    lines.append("【正解・根拠】")
    lines.append(result.correct_answer)
    lines.append("")

    lines.append("【重要語彙】")
    for v in result.vocab_list:
        lines.append(f"- {v.term}（{v.meaning}）: {v.exam_point}")
    lines.append("")

    lines.append("【IT専門用語】")
    for t in result.tech_terms:
        lines.append(f"- {t.term}: {t.concept} ／ 試験対策: {t.exam_focus}")
    lines.append("")

    def section(title: str, examples) -> None:
        lines.append(f"【{title}】")
        for i, ex in enumerate(examples, 1):
            lines.append(f"{i}. {ex.en}")
            lines.append(f"   {ex.jp}")
            lines.append(f"   keywords: {format_keywords(ex.key_terms)}")
            for kw in ex.key_terms:
                kw = normalize_keyword(kw)
                lines.append(f"     - {kw.term}: {kw.meaning} ／ IT実務: {kw.usage_note}")
                if kw.collocations:
                    lines.append(f"       📌 コロケーション: {', '.join(kw.collocations)}")
                for n, me in enumerate(kw.mini_examples, 1):
                    lines.append(f"       📝 {n}. {me.en} / {me.ja}")
        lines.append("")

    section("実務例文10選", result.business_examples)
    section("ITジョーク10選", result.humor_examples)
    section("恋愛ウィット10選", result.romance_examples)

    return "\n".join(lines).encode("utf-8")


# ---------------------------------------------------------------------------
# 単語解説ビュー（URLクエリパラメータ ?word=... と同期する独立ページ）
# ---------------------------------------------------------------------------
def render_word_detail_view() -> None:
    term = st.session_state.active_word
    cache_key = term.lower()

    def _back_button(key: str) -> None:
        if st.button("⬅️ 単語一覧に戻る", key=key, use_container_width=True):
            clear_word_selection()

    with st.container(border=True):
        st.markdown(
            "<div style='font-size:0.8rem; opacity:0.65; letter-spacing:0.02em;'>"
            "📖 単語解説</div>"
            f"<div style='font-size:1.75rem; font-weight:800; line-height:1.3; "
            f"margin-top:0.15rem;'>{term}</div>",
            unsafe_allow_html=True,
        )
        st.caption(f"🔗 URL: `?word={term}` （リロード・このURLの共有でも同じ解説が復元されます）")

    _back_button("word_detail_back_top")
    st.divider()

    if cache_key not in st.session_state.quick_lookup_cache:
        with st.spinner(f'"{term}" を検索中...'):
            try:
                looked_up = lookup_keyword(
                    term, on_retry=_on_gemini_retry, api_key=get_effective_api_key()
                )
            except GeminiServiceError as exc:
                _display_gemini_error(exc)
                looked_up = None
        if looked_up is not None:
            st.session_state.quick_lookup_cache[cache_key] = looked_up

    if cache_key in st.session_state.quick_lookup_cache:
        kw = normalize_keyword(st.session_state.quick_lookup_cache[cache_key])
        with st.container(border=True):
            render_keyword_dict(kw, key_prefix="worddetail")
        if st.button(
            "🚀 この単語で30例文を生成して深掘り ➔", key="word_detail_derive", use_container_width=True
        ):
            # 深掘り学習(derive)は履歴ビューへ遷移するため、単語解説ビューの状態を先に解除しておく。
            st.session_state.active_word = None
            st.query_params.pop("word", None)
            start_from_keyword(kw.term)
    else:
        st.warning("この単語の情報を取得できませんでした。APIキーを確認のうえ、もう一度お試しください。")

    st.divider()
    _back_button("word_detail_back_bottom")


# ---------------------------------------------------------------------------
# メイン表示
# ---------------------------------------------------------------------------
if st.session_state.active_word:
    render_word_detail_view()
elif st.session_state.history:
    try:
        depth = len(st.session_state.history)
        current_topic = st.session_state.labels[-1] if st.session_state.labels else "—"

        with st.container(border=True):
            st.markdown(
                "<div style='font-size:0.8rem; opacity:0.65; letter-spacing:0.02em;'>"
                "🎯 現在の学習テーマ</div>"
                f"<div style='font-size:1.75rem; font-weight:800; line-height:1.3; "
                f"margin-top:0.15rem;'>{current_topic}</div>",
                unsafe_allow_html=True,
            )

        breadcrumb = " ➔ ".join(st.session_state.labels)
        st.caption(f"🧭 第{depth}階層 ｜ 学習の道のり: {breadcrumb}")

        current = st.session_state.history[-1]

        col_export1, col_export2 = st.columns(2)
        with col_export1:
            st.download_button(
                "📥 CSVでダウンロード",
                data=build_csv(current),
                file_name=f"security_plus_layer{depth}.csv",
                mime="text/csv",
            )
        with col_export2:
            st.download_button(
                "📥 テキストでダウンロード",
                data=build_txt(current, st.session_state.labels[-1]),
                file_name=f"security_plus_layer{depth}.txt",
                mime="text/plain",
            )

        is_derived = depth > 1 or st.session_state.first_layer_is_derived
        main_tab_label = "🔬 単語・概念の徹底深掘り解説" if is_derived else "📘 設問解説 & 語彙"

        tab_main, tab_biz, tab_humor, tab_romance = st.tabs([
            main_tab_label,
            "💼 実務例文 (10)",
            "☕ ITジョーク (10)",
            "💘 恋愛ウィット (10)",
        ])

        with tab_main:
            if is_derived:
                st.subheader(f"🔍「{st.session_state.labels[-1]}」を徹底解剖")
                st.markdown(f"**和訳・ニュアンス**\n\n{current.translation}")
                st.success(f"**核心的な意味・用法**\n\n{current.correct_answer}")
            else:
                st.subheader("📝 和訳 & 正解")
                st.markdown(current.translation)
                st.info(f"**正解・解説:** {current.correct_answer}")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🔤 関連する重要語彙" if is_derived else "🔤 重要語彙・構文")
                for v in current.vocab_list:
                    st.markdown(f"**{v.term}** ({v.meaning})\n- {v.exam_point}")
            with col2:
                st.subheader("💻 関連するIT専門概念" if is_derived else "💻 IT専門用語の概念")
                for t in current.tech_terms:
                    st.markdown(f"**{t.term}**\n- 概念: {t.concept}\n- 試験対策: {t.exam_focus}")

        KEYWORDS_PER_ROW = 4
        CATEGORY_LABELS = {"biz": "実務例文", "humor": "ITジョーク", "romance": "恋愛ウィット"}

        def render_keyword_accordions(key_terms, prefix: str, i: int) -> None:
            if not key_terms:
                return
            st.caption("🏷️ Keywords（クリックでクイック辞書を表示）")
            for row_start in range(0, len(key_terms), KEYWORDS_PER_ROW):
                row = key_terms[row_start : row_start + KEYWORDS_PER_ROW]
                cols = st.columns(len(row))
                for j, (col, raw_kw) in enumerate(zip(cols, row)):
                    kw = normalize_keyword(raw_kw)
                    with col:
                        with st.popover(f"🔑 {kw.term}", use_container_width=True):
                            render_keyword_dict(
                                kw, key_prefix=f"{prefix}_{i}_{row_start + j}", show_pin=False
                            )
                            st.caption("📌 ピン留めは「解説ページを開く」から利用できます。")
                            st.divider()
                            col_open, col_deep = st.columns(2)
                            with col_open:
                                if st.button(
                                    "📖 解説ページを開く",
                                    key=f"{prefix}_{i}_kw_open_{row_start + j}",
                                    use_container_width=True,
                                ):
                                    st.session_state.quick_lookup_cache[kw.term.lower()] = kw
                                    select_word(kw.term)
                            with col_deep:
                                if st.button(
                                    "🚀 30例文で深掘り",
                                    key=f"{prefix}_{i}_kw_{row_start + j}",
                                    use_container_width=True,
                                ):
                                    derive(kw.term, [kw.term], f"{prefix}_kw", f"{i}_{row_start + j}")

        def render_examples(examples, prefix):
            category_label = CATEGORY_LABELS.get(prefix, prefix)
            for i, ex in enumerate(examples):
                with st.container():
                    st.markdown(f"**{i + 1}. {ex.en}**")
                    st.caption(ex.jp)
                    render_keyword_accordions(ex.key_terms, prefix, i)
                    col_derive, col_pin = st.columns([3, 1])
                    with col_derive:
                        if st.button("この内容から派生学習 ➔（例文全体）", key=f"{prefix}_{i}"):
                            derive(
                                ex.en,
                                [normalize_keyword(kw).term for kw in ex.key_terms],
                                prefix,
                                i,
                            )
                    with col_pin:
                        render_pin_controls(
                            f"{category_label}{i + 1}",
                            en_line=ex.en,
                            ja_line=ex.jp,
                            key_prefix=f"{prefix}_{i}_example",
                        )
                    st.divider()

        with tab_biz:
            render_examples(current.business_examples, "biz")

        with tab_humor:
            render_examples(current.humor_examples, "humor")

        with tab_romance:
            render_examples(current.romance_examples, "romance")

    except Exception as exc:  # noqa: BLE001 - keep the app usable instead of a hard crash
        st.error(
            "画面の描画中に問題が発生しました。スキーマ更新前に生成された古いデータが "
            "セッションに残っているなど、想定外の状態の可能性があります。\n\n"
            f"詳細: {type(exc).__name__}: {exc}"
        )
        if st.button("🧹 セッション状態をクリアしてやり直す", type="primary"):
            st.session_state.history = []
            st.session_state.labels = []
            st.session_state.first_layer_is_derived = False
            st.session_state.active_word = None
            st.query_params.pop("word", None)
            st.rerun()

else:
    st.info("左のサイドバーから画像をアップロードするか、問題文を入力して「🚀 解析・例文生成」を押してください。")

    if st.session_state.quickstart_sample is None:
        st.session_state.quickstart_sample = sample_quickstart_keywords()

    st.caption(f"全{len(ALL_QUICKSTART_KEYWORDS)}語のプールから毎回ランダム抜粋。タップした単語で即座に学習開始。")

    start_col, shuffle_col = st.columns(2)
    with start_col:
        if st.button(
            "🎲 完全ランダムなお題で即スタート", type="primary", use_container_width=True
        ):
            start_from_keyword(random.choice(ALL_QUICKSTART_KEYWORDS))
    with shuffle_col:
        if st.button("🔄 単語群をシャッフル（全入れ替え）", use_container_width=True):
            st.session_state.quickstart_sample = sample_quickstart_keywords()

    keywords = st.session_state.quickstart_sample
    cols_per_row = 4
    for i in range(0, len(keywords), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, kw in enumerate(keywords[i : i + cols_per_row]):
            with cols[j]:
                if st.button(kw, key=f"quick_kw_{i + j}", use_container_width=True):
                    start_from_keyword(kw)
