import csv
import datetime as dt
import html
import io
import json
import os
import random
import re
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
    # ピン留め保存先のObsidian Vault名。空文字が既定値 -> obsidian://new URIで
    # vaultパラメータを省略し、Obsidian側の現在アクティブなVaultに保存する
    # （固定名を入れておくと存在しないVault名として"Vault not found"になるため）。
    st.session_state.obsidian_vault = ""
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
if "quiz_stats" not in st.session_state:
    # 復習クイズの成績。term(小文字) -> {"term", "correct", "incorrect"}。
    # セッションを通じて蓄積し、苦手単語の優先出題に使う。
    st.session_state.quiz_stats = {}
if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = []
if "quiz_index" not in st.session_state:
    st.session_state.quiz_index = 0
if "quiz_active" not in st.session_state:
    # True の間はメイン画面を復習クイズの専用ビューに切り替える
    # （単語解説ビューの active_word と同様の「モード切り替えフラグ」）。
    st.session_state.quiz_active = False


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


_TAG_WORD_RE = re.compile(r"[0-9A-Za-z぀-ヿ一-鿿]+")


def sanitize_obsidian_tag(text: str, max_len: int = 30) -> str:
    """Obsidianタグ(#tag)として安全な単一トークンに正規化する。

    空白・記号（#, /, 改行, 絵文字 等 Obsidianのタグ構文を壊しうる文字）をすべて取り除き、
    各単語の頭文字を大文字にして連結する
    （例: "Zero Trust" -> "ZeroTrust", "mitigate the risk of" -> "MitigateTheRiskOf"）。
    タグとして残せる文字が無い場合は空文字を返し、呼び出し側で除外できるようにする。
    """
    words = _TAG_WORD_RE.findall(text)
    if not words:
        return ""
    combined = "".join(w[:1].upper() + w[1:] for w in words)
    return combined[:max_len]


def format_pin_entry(
    emoji: str,
    label_ja: str,
    term: str,
    en_line: str,
    ja_line: str,
    tag: str,
    tag_terms: list[str] | None = None,
) -> str:
    """Obsidianへの保存・コピー用にMarkdown箇条書き1エントリを組み立てる。

    tag_terms は例文中で使われた単語・熟語（複数キーワード組み合わせ学習の場合は
    その全キーワード）を、重複除去のうえ #タグとして追加する。
    """
    date_str = dt.date.today().isoformat()
    lines = [f"- [{emoji} {label_ja}] {term} / {date_str}"]
    if en_line:
        lines.append(f"    - EN: {en_line}")
    if ja_line:
        lines.append(f"    - JA: {ja_line}")

    tag_line = f"    - Tags: #SecurityPlus #{tag}"
    seen = {tag.lower()}
    for t in tag_terms or []:
        sanitized = sanitize_obsidian_tag(t)
        if sanitized and sanitized.lower() not in seen:
            seen.add(sanitized.lower())
            tag_line += f" #{sanitized}"
    lines.append(tag_line)
    lines.append("")
    return "\n".join(lines)


def build_obsidian_new_uri(vault: str, note: str, content: str) -> str:
    """ネイティブの obsidian://new URIスキームで、既存ノート末尾への追記リンクを組み立てる。

    vault が空文字の場合は vault パラメータ自体を省略する。Obsidianはvaultパラメータが
    無いとき現在アクティブなVaultを対象にするため、固定のデフォルト名を送って存在しない
    Vaultとして「Vault not found」になるのを避けられる。
    各値は quote(..., safe="") で個別にパーセントエンコードしてから連結するため、
    スペースや `+` `&` `=` などの記号がVault名・本文に含まれていても壊れない。
    """
    params = {}
    if vault:
        params["vault"] = vault
    params["file"] = note
    params["content"] = content
    params["append"] = "true"
    query = "&".join(f"{k}={quote(v, safe='')}" for k, v in params.items())
    return f"obsidian://new?{query}"


def render_pin_controls(
    term: str,
    en_line: str,
    ja_line: str,
    key_prefix: str,
    tag_terms: list[str] | None = None,
) -> None:
    """3段階の重要度でObsidianへワンクリック追記保存 + Markdownコピーを提供する。

    tag_terms を渡すと、#SecurityPlus #P1 等の既定タグに加えて、その単語・熟語
    （複数キーワード組み合わせ学習の場合は関与した全キーワード）も #タグとして追加される。
    """
    vault = st.session_state.obsidian_vault.strip()
    with st.popover("📌 ピン留め / Obsidian保存", use_container_width=False, key=f"{key_prefix}_pin_popover"):
        vault_desc = f"**{vault}**" if vault else "現在開いている**アクティブなVault**"
        st.caption(f"保存先: {vault_desc} ／ `{OBSIDIAN_NOTE_NAME}.md`（末尾に追記）")
        entries = {
            label_ja: format_pin_entry(emoji, label_ja, term, en_line, ja_line, tag, tag_terms)
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
            tag_terms=[kw.term],
        )


# ---------------------------------------------------------------------------
# 復習クイズ: これまでに生成済みの語彙データ（vocab_list/tech_terms/key_terms/
# クイック検索結果）を素材に、追加のGemini呼び出しなしでその場に4択問題を組み立てる。
# ---------------------------------------------------------------------------
QUIZ_CHOICE_TEXT_LIMIT = 70


class QuizTerm:
    __slots__ = ("term", "meaning")

    def __init__(self, term: str, meaning: str) -> None:
        self.term = term
        self.meaning = meaning


def _truncate_for_quiz(text: str, limit: int = QUIZ_CHOICE_TEXT_LIMIT) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def collect_quiz_pool() -> dict[str, QuizTerm]:
    """セッション内でこれまでに学習した単語・熟語を term(小文字) -> QuizTerm でまとめる。

    同じ単語が複数回登場した場合は最初に見つかったものを採用する。
    """
    pool: dict[str, QuizTerm] = {}

    def add(term: str, meaning: str) -> None:
        term = term.strip()
        meaning = meaning.strip()
        if not term or not meaning:
            return
        key = term.lower()
        if key not in pool:
            pool[key] = QuizTerm(term=term, meaning=_truncate_for_quiz(meaning))

    for result in st.session_state.history:
        for v in result.vocab_list:
            add(v.term, v.meaning)
        for t in result.tech_terms:
            add(t.term, t.concept)
        for ex in result.business_examples + result.humor_examples + result.romance_examples:
            for kw in ex.key_terms:
                kw = normalize_keyword(kw)
                add(kw.term, kw.meaning)

    for kw in st.session_state.quick_lookup_cache.values():
        kw = normalize_keyword(kw)
        add(kw.term, kw.meaning)

    return pool


def build_quiz_questions(pool: dict[str, QuizTerm], n: int, weak_only: bool = False) -> list[dict]:
    """pool から n 問の4択問題を組み立てる。

    各問題は「単語→意味」「意味→単語」のどちらかをランダムに出題し、不正解の選択肢は
    プール内の他の単語からランダムに抽出する（プールが小さい場合は選択肢数を減らす）。
    """
    candidates = list(pool.values())
    if weak_only:
        weak_terms = [
            pool[key]
            for key, s in st.session_state.quiz_stats.items()
            if s["incorrect"] > 0 and key in pool
        ]
        if len(weak_terms) >= 2:
            candidates = weak_terms

    random.shuffle(candidates)
    n = min(n, len(candidates))
    questions: list[dict] = []
    for qt in candidates[:n]:
        others = [t for t in pool.values() if t.term.lower() != qt.term.lower()]
        distractors = random.sample(others, k=min(3, len(others)))
        direction = random.choice(["term_to_meaning", "meaning_to_term"])
        if direction == "term_to_meaning":
            prompt = f'🔤 「{qt.term}」の意味・用法として最も適切なものは？'
            correct_choice = qt.meaning
            choices = [qt.meaning] + [d.meaning for d in distractors]
        else:
            prompt = f'📖 次の意味・用法に当てはまる単語・熟語はどれ？\n\n"{qt.meaning}"'
            correct_choice = qt.term
            choices = [qt.term] + [d.term for d in distractors]
        random.shuffle(choices)
        questions.append(
            {
                "term_key": qt.term.lower(),
                "term": qt.term,
                "prompt": prompt,
                "choices": choices,
                "correct_index": choices.index(correct_choice),
                "answered": False,
                "selected_index": None,
            }
        )
    return questions


def _record_quiz_answer(term_key: str, term: str, is_correct: bool) -> None:
    stats = st.session_state.quiz_stats.setdefault(term_key, {"term": term, "correct": 0, "incorrect": 0})
    if is_correct:
        stats["correct"] += 1
    else:
        stats["incorrect"] += 1


def start_quiz(n: int, weak_only: bool = False) -> None:
    pool = collect_quiz_pool()
    st.session_state.quiz_questions = build_quiz_questions(pool, n, weak_only=weak_only)
    st.session_state.quiz_index = 0
    st.session_state.quiz_active = True
    st.session_state.active_word = None
    st.query_params.pop("word", None)
    st.rerun()


def exit_quiz() -> None:
    st.session_state.quiz_active = False
    st.session_state.quiz_questions = []
    st.session_state.quiz_index = 0
    st.rerun()


def derive(seed_text: str, key_terms: list[str], mode: str = "derived") -> None:
    seed_prompt = f"Target: {seed_text}\nKey Terms: {', '.join(key_terms)}"
    with st.spinner("派生学習コンテンツを生成中..."):
        try:
            new_result = analyze_and_generate(
                seed_prompt,
                mode=mode,
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
    derive(keyword, [keyword])


# ---------------------------------------------------------------------------
# 複数キーワード組み合わせ学習（用語×用語・用語×熟語）
#
# "Zero Trust, Microsegmentation" のようにカンマ等で明示的に区切られた入力のみを
# 複数キーワードとして分割する。区切り文字の無い自由記述（例:"revoke a certificate"
# のような、スペースを含む既存の単一熟語）を誤って分割すると、サイト全体で使われている
# 単一熟語ルックアップ（例文中のキーワードpopover等）を壊してしまうため、
# 意図的に「区切り文字が無ければ分割しない」設計にしている。
# ---------------------------------------------------------------------------
MAX_COMBO_KEYWORDS = 5
_COMBO_DELIMITER_RE = re.compile(
    r"\s*,\s*|\s*、\s*|\s*;\s*|\s*；\s*|\s*&\s*|\n+|\s+/\s+|\s+／\s+|\s+[x×]\s+",
    re.IGNORECASE,
)


def split_combo_keywords(raw_text: str) -> list[str]:
    """カンマ・スラッシュ(空白で挟まれた場合のみ)・"&"・"×"/"x" などの明示的な区切り文字が
    ある場合のみ、複数キーワードとして分割する。区切りが無ければ元の文字列を1件のまま返す。
    """
    parts = [p.strip() for p in _COMBO_DELIMITER_RE.split(raw_text) if p.strip()]
    if len(parts) < 2:
        return [raw_text.strip()]
    return parts[:MAX_COMBO_KEYWORDS]


def looks_like_keyword_combo(raw_text: str, parts: list[str]) -> bool:
    """メイン入力欄向けの安全策。実際の設問文（"?"を含む・断片が長い 等）を誤って
    複数キーワードと誤認しないよう、断片がいずれも単語・熟語らしい短さの場合のみ
    複数キーワード入力とみなす。
    """
    if len(parts) < 2 or "?" in raw_text:
        return False
    return all(len(p) <= 40 and len(p.split()) <= 6 for p in parts)


def start_combo(keywords: list[str]) -> None:
    """明示的な区切り文字で分割された複数キーワード（用語×用語・用語×熟語）の
    組み合わせ学習を、現在の学習階層に積んで開始する。"""
    label = " × ".join(keywords)
    st.session_state.first_layer_is_derived = True
    st.session_state.active_word = None
    st.query_params.pop("word", None)
    derive(label, keywords, mode="combo")


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
        combo_parts = split_combo_keywords(text_input) if text_input else [text_input]
        is_combo_input = not uploaded_file and looks_like_keyword_combo(text_input, combo_parts)
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
                elif is_combo_input:
                    result = analyze_and_generate(
                        f"Target: {' × '.join(combo_parts)}\nKey Terms: {', '.join(combo_parts)}",
                        mode="combo",
                        on_retry=_on_gemini_retry,
                        api_key=get_effective_api_key(),
                    )
                    label = " × ".join(combo_parts)[:24]
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
        st.session_state.first_layer_is_derived = is_combo_input
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
    st.caption(
        "1語なら「単語解説」ビューへ。カンマ・スラッシュ・&・×等で区切って2語以上入力すると、"
        "その組み合わせを使った例文・解説（用語×用語・用語×熟語）を生成します。"
    )
    lookup_term = st.text_input(
        "単語・熟語を入力（例: Zero Trust, Microsegmentation）", key="quick_lookup_input"
    )
    if st.button("調べる", key="quick_lookup_button"):
        raw_term = lookup_term.strip()
        if not raw_term:
            st.warning("単語・熟語を入力してください。")
        else:
            combo_keywords = split_combo_keywords(raw_term)
            if len(combo_keywords) >= 2:
                start_combo(combo_keywords)
            else:
                term = combo_keywords[0]
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
    st.subheader("🧠 復習クイズ")
    _quiz_pool = collect_quiz_pool()
    _weak_count = sum(1 for s in st.session_state.quiz_stats.values() if s["incorrect"] > 0)
    st.caption(f"学習済み単語: {len(_quiz_pool)}語 ／ 苦手: {_weak_count}語")

    if len(_quiz_pool) < 2:
        st.caption("単語を2つ以上学習すると、ここから復習クイズに挑戦できます。")
    else:
        _quiz_max = min(20, len(_quiz_pool))
        quiz_len = st.slider(
            "問題数", min_value=1, max_value=_quiz_max, value=min(10, _quiz_max), key="quiz_len_slider"
        )
        quiz_weak_only = st.checkbox(
            "苦手な単語を優先して出題",
            key="quiz_weak_only",
            help="過去に間違えたことのある単語が2語以上あれば、それらを中心に出題します。",
        )
        if st.button("🚀 クイズを開始", type="primary", use_container_width=True):
            start_quiz(quiz_len, weak_only=quiz_weak_only)

    if st.session_state.quiz_stats:
        with st.expander("📊 苦手単語ランキング"):
            ranked = sorted(
                st.session_state.quiz_stats.values(),
                key=lambda s: s["correct"] / max(s["correct"] + s["incorrect"], 1),
            )
            for s in ranked[:10]:
                attempts = s["correct"] + s["incorrect"]
                accuracy = (s["correct"] / attempts * 100) if attempts else 0
                st.caption(f"- **{s['term']}**: 正答率 {accuracy:.0f}%（{s['correct']}/{attempts}）")
            st.divider()
            if st.button("🧹 クイズ成績をリセット", key="quiz_stats_reset"):
                st.session_state.quiz_stats = {}
                st.rerun()

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
            "空欄のままなら、現在Obsidianで開いているアクティブなVaultに保存されます。"
        )
        st.text_input(
            "Vault名（空欄可）",
            key="obsidian_vault",
            placeholder="空欄 = 現在開いているVault",
        )

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
            f"margin-top:0.15rem;'>{html.escape(term)}</div>",
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
# 復習クイズビュー（単語解説ビューと同様、フラグが立っている間はメイン画面を占有する）
# ---------------------------------------------------------------------------
def render_quiz_view() -> None:
    questions = st.session_state.quiz_questions
    idx = st.session_state.quiz_index
    total = len(questions)

    if idx >= total:
        correct_count = sum(
            1 for q in questions if q["answered"] and q["selected_index"] == q["correct_index"]
        )
        with st.container(border=True):
            st.markdown("### 🏁 クイズ結果")
            st.markdown(f"## {correct_count} / {total} 問正解")
            missed = [q["term"] for q in questions if q["answered"] and q["selected_index"] != q["correct_index"]]
            if missed:
                st.markdown("**苦手として記録した単語:**")
                st.markdown(", ".join(missed))
            else:
                st.success("全問正解です！お見事！")

        col_retry, col_exit = st.columns(2)
        with col_retry:
            if st.button("🔁 もう一度挑戦", use_container_width=True):
                start_quiz(total)
        with col_exit:
            if st.button("⬅️ 学習に戻る", type="primary", use_container_width=True):
                exit_quiz()
        return

    q = questions[idx]
    st.progress(idx / total, text=f"問題 {idx + 1} / {total}")

    with st.container(border=True):
        st.markdown(f"### {q['prompt']}")
        if not q["answered"]:
            for i, choice in enumerate(q["choices"]):
                if st.button(choice, key=f"quiz_choice_{idx}_{i}", use_container_width=True):
                    q["answered"] = True
                    q["selected_index"] = i
                    _record_quiz_answer(q["term_key"], q["term"], i == q["correct_index"])
                    st.rerun()
        else:
            for i, choice in enumerate(q["choices"]):
                if i == q["correct_index"]:
                    st.success(f"✅ {choice}")
                elif i == q["selected_index"]:
                    st.error(f"❌ {choice}")
                else:
                    st.markdown(f"- {choice}")

            if q["selected_index"] == q["correct_index"]:
                st.success("正解です！")
            else:
                st.warning(f"不正解…　正解は「{q['choices'][q['correct_index']]}」でした。")

            if st.button("次の問題へ ➔", type="primary", use_container_width=True, key=f"quiz_next_{idx}"):
                st.session_state.quiz_index += 1
                st.rerun()

    if st.button("⏹️ クイズを中断して学習に戻る", key=f"quiz_abort_{idx}"):
        exit_quiz()


# ---------------------------------------------------------------------------
# メイン表示
# ---------------------------------------------------------------------------
if st.session_state.quiz_active:
    render_quiz_view()
elif st.session_state.active_word:
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
                f"margin-top:0.15rem;'>{html.escape(current_topic)}</div>",
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
                                    derive(kw.term, [kw.term])

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
                            )
                    with col_pin:
                        render_pin_controls(
                            f"{category_label}{i + 1}",
                            en_line=ex.en,
                            ja_line=ex.jp,
                            key_prefix=f"{prefix}_{i}_example",
                            tag_terms=[normalize_keyword(kw).term for kw in ex.key_terms],
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
