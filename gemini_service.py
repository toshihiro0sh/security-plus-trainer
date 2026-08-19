import logging
import os
import random
import traceback

import httpx
from PIL import Image
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, Field
from tenacity import Retrying, retry_if_exception, wait_exponential

logger = logging.getLogger("gemini_service")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)

# HTTPリクエストのタイムアウト(ミリ秒)。未設定だとGemini側が無応答のまま固まった場合に
# 無期限に待機し続けてしまうため、必ず上限を設ける。
# GENERATE_TIMEOUT_MS: 30例文フル生成は観測上おおよそ60〜90秒かかるため、十分な余裕を持たせる。
# LOOKUP_TIMEOUT_MS: 単語1つの軽量ルックアップ用。
GENERATE_TIMEOUT_MS = 150_000
LOOKUP_TIMEOUT_MS = 60_000


def _is_retryable_error(exc: BaseException) -> bool:
    """503(UNAVAILABLE)・429(混雑/レート制限)・一時的な通信エラーのみリトライ対象とする。"""
    if isinstance(exc, genai_errors.ServerError):  # 5xx (503 UNAVAILABLE 含む)
        return True
    if isinstance(exc, genai_errors.ClientError) and getattr(exc, "code", None) == 429:
        return True
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError)):
        return True
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    return False


def _is_rate_limit_error(exc: BaseException) -> bool:
    """429 RESOURCE_EXHAUSTED（クォータ/レートリミット超過）かどうか。"""
    return isinstance(exc, genai_errors.ClientError) and getattr(exc, "code", None) == 429


def _is_timeout_error(exc: BaseException) -> bool:
    """設定したタイムアウト秒数に達し、Gemini APIから応答が得られなかったかどうか。"""
    return isinstance(exc, (httpx.TimeoutException, TimeoutError))


def _extract_retry_delay_seconds(exc: BaseException) -> float | None:
    """429エラーのレスポンス本文から google.rpc.RetryInfo.retryDelay (例: "31s") を取り出す。"""
    body = getattr(exc, "details", None)
    if not isinstance(body, dict):
        return None
    candidates = [body]
    inner = body.get("error")
    if isinstance(inner, dict):
        candidates.append(inner)
    for candidate in candidates:
        for item in candidate.get("details", []) or []:
            if isinstance(item, dict) and item.get("retryDelay"):
                raw = str(item["retryDelay"]).rstrip("s")
                try:
                    return float(raw)
                except ValueError:
                    continue
    return None


_FALLBACK_WAIT = wait_exponential(multiplier=2, min=2, max=8)  # 503等: 2s -> 4s -> 8s


def _gemini_wait(retry_state) -> float:
    """429の場合はレスポンスのretryDelay(なければ15〜30秒)、それ以外は指数バックオフ。"""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if exc is not None and _is_rate_limit_error(exc):
        delay = _extract_retry_delay_seconds(exc)
        if delay is not None:
            return max(delay, 1.0)
        return random.uniform(15, 30)
    return _FALLBACK_WAIT(retry_state)


def _gemini_stop(retry_state) -> bool:
    """タイムアウトは1回分の待機時間がすでに長いため、無駄な長時間待機を避けて
    最大2回(初回+1回)までに抑える。503等の一時的なエラーは従来通り最大4回まで許容する。"""
    exc = retry_state.outcome.exception() if retry_state.outcome and retry_state.outcome.failed else None
    max_attempts = 2 if exc is not None and _is_timeout_error(exc) else 4
    return retry_state.attempt_number >= max_attempts


def _make_before_sleep(on_retry):
    def _before_sleep(retry_state) -> None:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        sleep_seconds = getattr(retry_state.next_action, "sleep", None)
        is_rate_limit = exc is not None and _is_rate_limit_error(exc)
        is_timeout = exc is not None and _is_timeout_error(exc)
        wait_str = f"{sleep_seconds:.0f}" if sleep_seconds is not None else "?"
        if is_rate_limit:
            logger.warning(
                "Gemini APIのレート制限(429 RESOURCE_EXHAUSTED)に達しました "
                "(試行%d回目失敗、%s秒待機して自動リトライします): %r",
                retry_state.attempt_number,
                wait_str,
                exc,
            )
        elif is_timeout:
            logger.warning(
                "Gemini APIの応答がタイムアウトしました "
                "(試行%d回目失敗、%s秒待機して再試行します): %r",
                retry_state.attempt_number,
                wait_str,
                exc,
            )
        else:
            logger.warning(
                "Gemini API呼び出しでリトライ可能なエラーが発生しました "
                "(試行%d回目失敗、%s秒待機してリトライします): %r",
                retry_state.attempt_number,
                wait_str,
                exc,
            )
        if on_retry is not None:
            try:
                on_retry(
                    {
                        "is_rate_limit": is_rate_limit,
                        "is_timeout": is_timeout,
                        "attempt": retry_state.attempt_number,
                        "wait_seconds": sleep_seconds,
                        "error": exc,
                    }
                )
            except Exception:  # noqa: BLE001 - UI通知の失敗でリトライ自体は止めない
                logger.exception("on_retry コールバックの実行中にエラーが発生しました。")

    return _before_sleep


def _call_generate_content(client: genai.Client, *, on_retry=None, **kwargs):
    retrying = Retrying(
        retry=retry_if_exception(_is_retryable_error),
        stop=_gemini_stop,  # 通常は初回+最大3回、タイムアウトは初回+最大1回
        wait=_gemini_wait,
        before_sleep=_make_before_sleep(on_retry),
        reraise=True,
    )
    return retrying(client.models.generate_content, **kwargs)


# NOTE: response_schema に渡す Pydantic モデルは、Google GenAI SDK が対応する
# フィールドのみで構成すること。min_length/max_length など pydantic v2 の
# 追加制約(annotated_types 経由)はスキーマ変換時に正しく解釈されず、
# Gemini API から 400 INVALID_ARGUMENT が返る原因になるため使用しない。
# 個数の指定はすべてプロンプト（description / システム指示）側で行う。
class MiniExampleItem(BaseModel):
    en: str = Field(description="実務に直結する短い英語例文")
    ja: str = Field(description="日本語訳")


class KeywordDictItem(BaseModel):
    term: str = Field(description="単語または熟語")
    meaning: str = Field(description="一般的な意味・品詞")
    usage_note: str = Field(description="IT / Security+実務での用法・着眼点")
    collocations: list[str] = Field(
        description=(
            "実務で頻出する熟語・コロケーションをちょうど3〜5個。"
            '例: ["digital certificate", "revoke a certificate", "certificate authority"]'
        )
    )
    mini_examples: list[MiniExampleItem] = Field(
        description="この単語・熟語を使った実務に直結する短文の英語例文と日本語訳のペアをちょうど5組"
    )


class ExampleItem(BaseModel):
    en: str = Field(description="英語例文")
    jp: str = Field(description="日本語訳")
    key_terms: list[KeywordDictItem] = Field(
        description=(
            "例文中で実際に使用した重要キーワード（単語・熟語・実務複合語）をちょうど3〜6個、"
            "クイック辞書情報付きで網羅的に抽出したもの"
        )
    )


class VocabItem(BaseModel):
    term: str = Field(description="単語・熟語")
    meaning: str = Field(description="意味・品詞")
    exam_point: str = Field(description="英検2級〜準1級目線での着眼点・多義性")


class TechTermItem(BaseModel):
    term: str = Field(description="IT専門用語")
    concept: str = Field(description="IT初学者向けのかみ砕いた概念解説")
    exam_focus: str = Field(description="Security+試験での狙われどころ")


class SecurityPlusResponse(BaseModel):
    translation: str = Field(description="問題文および選択肢の和訳")
    correct_answer: str = Field(description="正解とその技術的根拠")
    vocab_list: list[VocabItem] = Field(description="重要語彙リスト")
    tech_terms: list[TechTermItem] = Field(description="IT専門用語解説")
    business_examples: list[ExampleItem] = Field(description="実務例文10選")
    humor_examples: list[ExampleItem] = Field(description="IT現場ジョーク10選")
    romance_examples: list[ExampleItem] = Field(description="恋愛ウィット・口説き文句10選")


class GeminiServiceError(RuntimeError):
    """Gemini API呼び出し・応答パースに関するエラー"""


def _get_client(api_key: str | None = None) -> genai.Client:
    """優先度1: 呼び出し側から渡された(ユーザー個別の)キー。
    優先度2/3(共有デフォルトキー: Streamlit Secrets / .env)は、呼び出し側が
    あらかじめ os.environ["GEMINI_API_KEY"] に解決済みであることを前提にフォールバックする。
    """
    resolved = api_key or os.environ.get("GEMINI_API_KEY")
    if not resolved:
        raise GeminiServiceError(
            "GEMINI_API_KEY が設定されていません。サイドバーからご自身のAPIキーを入力するか、"
            ".env / Streamlit Secrets に GEMINI_API_KEY を設定してください。"
        )
    return genai.Client(api_key=resolved)


def is_rate_limit_error(exc: BaseException) -> bool:
    """GeminiServiceError（または元例外）が429 RESOURCE_EXHAUSTEDに起因するかどうか。"""
    if _is_rate_limit_error(exc):
        return True
    cause = getattr(exc, "__cause__", None)
    return cause is not None and _is_rate_limit_error(cause)


def is_timeout_error(exc: BaseException) -> bool:
    """GeminiServiceError（または元例外）がタイムアウトに起因するかどうか。"""
    if _is_timeout_error(exc):
        return True
    cause = getattr(exc, "__cause__", None)
    return cause is not None and _is_timeout_error(cause)


KEY_TERM_INSTRUCTIONS = (
    "各例文の key_terms には、その例文の中でCompTIA Security+受験者・ITエンジニアが"
    "着目すべき重要な単語・熟語・実務複合語を3〜6個、網羅的に抽出してください。"
    "immediately のような一般的な副詞・機能語だけに偏ってはいけません。"
    "セキュリティ技術名詞（例: certificate, encryption, repository, unencrypted）、"
    "プロトコル・規格名、実務運用プロセス（例: employee offboarding, incident response）、"
    "重要動詞（例: revoke, mitigate, escalate）を優先的に拾ってください。"
    "各キーワードについて、以下をすべて埋めてクイック辞書として記載してください:\n"
    "- term: 単語・熟語\n"
    "- meaning: 一般的な意味・品詞\n"
    "- usage_note: IT / Security+実務での用法・着眼点\n"
    "- collocations: この語と一緒に実務でよく使われる熟語・コロケーションを3〜5個"
    "（例: revoke -> ['revoke a certificate', 'revoke access', 'revoke privileges']）\n"
    "- mini_examples: この語を使った実務に直結する短い英語例文とその日本語訳のペアをちょうど5組"
)


def _generate_structured(
    client,
    contents,
    system_prompt,
    response_model,
    *,
    log_context: str,
    on_retry=None,
    timeout_ms: int = GENERATE_TIMEOUT_MS,
):
    try:
        response = _call_generate_content(
            client,
            on_retry=on_retry,
            model="gemini-3.6-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=response_model,
                temperature=0.7,
                http_options=types.HttpOptions(timeout=timeout_ms),
            ),
        )
    except Exception as exc:  # noqa: BLE001 - surface any API/network error to the UI
        logger.error(
            "Gemini API呼び出しに失敗しました (%s)。リトライ上限に達したか、"
            "リトライ対象外のエラーです。",
            log_context,
        )
        logger.error("例外: %r", exc)
        # google-genai の例外は詳細なエラー本文を response/body/args に持つことがあるため
        # 可能な限り拾ってターミナルに出す。
        for attr in ("response", "body", "details"):
            detail = getattr(exc, attr, None)
            if detail:
                logger.error("詳細(%s): %s", attr, detail)
        logger.error(traceback.format_exc())
        if _is_timeout_error(exc):
            timeout_seconds = timeout_ms / 1000
            raise GeminiServiceError(
                f"⏱️ Gemini APIから{timeout_seconds:.0f}秒待っても応答がありませんでした"
                "（サーバー混雑または通信状況が原因の可能性があります）。"
                "しばらく待ってから、もう一度実行してください。"
            ) from exc
        raise GeminiServiceError(f"Gemini API呼び出しに失敗しました: {exc}") from exc

    if not response.text:
        logger.error("Gemini APIから空の応答が返されました (%s)。response=%r", log_context, response)
        raise GeminiServiceError("Gemini APIから空の応答が返されました。")

    try:
        return response_model.model_validate_json(response.text)
    except Exception as exc:  # noqa: BLE001
        logger.error("Gemini応答のJSON解析に失敗しました (%s): %r", log_context, exc)
        logger.error("受信した応答本文:\n%s", response.text)
        logger.error(traceback.format_exc())
        raise GeminiServiceError(f"Gemini応答のJSON解析に失敗しました: {exc}") from exc


def analyze_and_generate(
    input_data,
    is_image: bool = False,
    mode: str = "question",
    on_retry=None,
    api_key: str | None = None,
) -> SecurityPlusResponse:
    client = _get_client(api_key)

    if mode == "derived":
        system_prompt = (
            "あなたはCompTIA Security+の専任指導教官および英語メンターです。"
            "学習者は特定の単語・表現・例文にさらに興味を持ち、そこを起点とした深掘り学習を求めています。"
            "入力されたターゲット語句・例文を単独の学習トピックとして詳細に分析し、指定されたスキーマに従ってJSONを出力してください。"
            "translation にはターゲットの自然な和訳・ニュアンスを、correct_answer には設問の正解ではなく"
            "このターゲット語句・表現の核心的な意味/用法/紛らわしい表現との違いを詳しく記載してください。"
            "vocab_list と tech_terms は、このターゲットに関連する周辺語彙・概念を新たに掘り下げて生成してください。"
            "各例文カテゴリ（実務、ITジョーク、恋愛ウィット）には必ず10個ずつ、このターゲットおよび関連語彙を"
            "活用した高品質な例文を生成してください。\n"
            f"{KEY_TERM_INSTRUCTIONS}"
        )
    else:
        system_prompt = (
            "あなたはCompTIA Security+の専任指導教官および英語メンターです。"
            "入力された設問またはテキストを詳細に分析し、指定されたスキーマに従ってJSONを出力してください。"
            "各例文カテゴリ（実務、ITジョーク、恋愛ウィット）には必ず10個ずつの高品質な例文を生成してください。"
            "例文はいずれも vocab_list や tech_terms で抽出した語彙・専門用語を積極的に活用してください。\n"
            f"{KEY_TERM_INSTRUCTIONS}"
        )

    if is_image:
        contents = [input_data, "この画像の設問を分析してください。"]
    else:
        contents = [input_data]

    return _generate_structured(
        client,
        contents,
        system_prompt,
        SecurityPlusResponse,
        log_context=f"analyze_and_generate mode={mode} is_image={is_image}",
        on_retry=on_retry,
    )


LOOKUP_SYSTEM_PROMPT = (
    "あなたはCompTIA Security+の専任指導教官および英語メンターです。"
    "学習者がサイドバーのクイック辞書検索で単語・熟語を1つ調べようとしています。"
    "入力された語をIT / Security+実務の文脈でクイック辞書として解説し、"
    "指定されたスキーマに従ってJSONを出力してください。\n"
    "- term: 検索された単語・熟語（自然な形に正規化してよい）\n"
    "- meaning: 一般的な意味・品詞\n"
    "- usage_note: IT / Security+実務での用法・着眼点\n"
    "- collocations: この語と一緒に実務でよく使われる熟語・コロケーションを3〜5個\n"
    "- mini_examples: この語を使った実務に直結する短い英語例文とその日本語訳のペアをちょうど5組"
)


def lookup_keyword(term: str, on_retry=None, api_key: str | None = None) -> KeywordDictItem:
    """サイドバーのクイック辞書検索用に、単語1つを軽量にルックアップする。"""
    client = _get_client(api_key)
    contents = [f'Look up this word or phrase for a CompTIA Security+ learner: "{term}"']
    return _generate_structured(
        client,
        contents,
        LOOKUP_SYSTEM_PROMPT,
        KeywordDictItem,
        log_context=f"lookup_keyword term={term!r}",
        on_retry=on_retry,
        timeout_ms=LOOKUP_TIMEOUT_MS,
    )
