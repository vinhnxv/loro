"""Seed LanguageProfile instances: tier-1 (EN/VI/FR/ES) + the generic fallback.

Seed constants are starting points; tier-1 CPS/rate/tolerance are calibrated from
real FR/ES runs in U11 (the registry test asserts they stay non-placeholder).
The VI profile is NOT a seed in that sense — it replicates the legacy syllable
model (4.3 syl/s) and the exact translate SYSTEM prompt verbatim so the VI/EN
default stays byte-identical at the fingerprint level (KTD2, R19). External
grounding for the non-VI rates/expansions: Netflix Timed Text (Latin ~17 CPS),
Coupé et al. 2019 (speech rates VI ~5.3 / FR ~6.9 / ES ~7.8 syl/s), and standard
localization expansion factors (FR +15-20%, ES +15-30%).
"""

from loro.profiles.base import (
    ContextLabels, LanguageProfile, char_count, vi_syllable_count)

# The legacy Vietnamese dubbing system prompt, byte-identical to the historical
# `nodes.translate.SYSTEM`. Kept verbatim here so the VI translate fingerprint
# (which folds the system prompt) is unchanged when U8 sources it from the
# profile. A profiles-registry test asserts this equals `translate.SYSTEM`.
_VI_SYSTEM = (
    "Bạn là biên dịch viên lồng tiếng chuyên nghiệp Anh-Việt. Dịch thoại sang "
    "tiếng Việt tự nhiên, văn nói, giữ đúng ý và sắc thái. Mỗi câu phải đọc vừa "
    "trong thời lượng cho phép: tôn trọng giới hạn số âm tiết của từng câu, "
    "rút gọn thay vì dịch sát chữ khi cần. Giữ nguyên thuật ngữ tiếng Anh là "
    "từ mượn thông dụng trong ngành của video (dựa vào Bối cảnh video — ví dụ "
    "video công nghệ giữ nguyên 'agent', 'model', 'deploy'); không ép dịch "
    "thuật ngữ chuyên ngành thành tiếng Việt tối nghĩa. Không thêm chú thích."
)

_FR_SYSTEM = (
    "Vous êtes un traducteur professionnel de doublage. Traduisez les répliques "
    "en français naturel et parlé, en gardant le sens et le ton. Chaque réplique "
    "doit pouvoir se dire dans la durée impartie : respectez la limite de longueur "
    "de chaque ligne, condensez plutôt que de traduire mot à mot si nécessaire. "
    "Conservez les termes anglais qui sont des emprunts courants dans le domaine "
    "de la vidéo (d'après le Contexte vidéo — p. ex. une vidéo tech garde « agent », "
    "« model », « deploy ») ; ne forcez pas la traduction de termes techniques en "
    "un français obscur. N'ajoutez aucune note."
)

_ES_SYSTEM = (
    "Eres un traductor profesional de doblaje. Traduce los diálogos a un español "
    "natural y hablado, conservando el sentido y el matiz. Cada línea debe poder "
    "decirse dentro de la duración permitida: respeta el límite de longitud de cada "
    "frase, condensa en lugar de traducir literalmente cuando sea necesario. "
    "Conserva los términos en inglés que sean préstamos habituales en el ámbito del "
    "vídeo (según el Contexto del vídeo — p. ej. un vídeo de tecnología mantiene "
    "«agent», «model», «deploy»); no fuerces la traducción de términos técnicos a un "
    "español confuso. No añadas notas."
)

_EN_SYSTEM = (
    "You are a professional dubbing translator. Render the dialogue into natural, "
    "spoken English, preserving meaning and tone. Each line must be sayable within "
    "its allotted duration: respect each line's length limit, condensing rather than "
    "translating word-for-word when needed. Keep source-language terms that are "
    "common loanwords in the video's domain (per the Video context). Do not add notes."
)

_GENERIC_SYSTEM = (
    "You are a professional dubbing translator. Translate the dialogue into the "
    "target language as natural, spoken speech, preserving meaning and tone. Each "
    "line must be sayable within its allotted duration: respect each line's length "
    "limit, condensing rather than translating word-for-word when needed. Keep "
    "source-language terms that are common loanwords in the video's domain. Do not "
    "add notes."
)

# The historical default preset pool (Soniox), shared by the Latin tier-1 profiles
# whose Soniox voices are timbre presets steered to a language by `tts_language_code`.
_SONIOX_POOL = ("Adrian", "Maya", "Noah", "Nina", "Jack", "Emma")

# The exact legacy Vietnamese context labels + per-line directive, byte-identical
# to the historical translate node so the VI sent prompt is unchanged (the
# fingerprint excludes these, but the VI prompt-content tests assert them).
_VI_LABELS = ContextLabels(
    video_context="Bối cảnh video",
    shot_visuals="Mô tả hình ảnh đoạn này",
    summary="Tóm tắt mạch nội dung tới đây",
    neighbors_before="Câu liền trước (gốc tiếng Anh)",
    neighbors_after="Câu liền sau (gốc tiếng Anh)",
    prev_translations="Bản dịch các câu ngay trước (giữ nhất quán đại từ/độ trang "
                      "trọng/thuật ngữ, KHÔNG dịch lại)",
    instruction='Dịch các câu thoại sau. Trả về DUY NHẤT một JSON array dạng '
                '[{"i": <số>, "vi": "<bản dịch>"}] với đúng các chỉ số i đã cho.',
)

# English label set shared by every non-VI profile: the LLM reads English labels
# around any-language content fine, and the target language is established by the
# (target-language) system prompt. The line objects carry "src"/"budget".
_EN_LABELS = ContextLabels(
    video_context="Video context",
    shot_visuals="On-screen visuals here",
    summary="Story so far",
    neighbors_before="Preceding lines (source original)",
    neighbors_after="Following lines (source original)",
    prev_translations="Translations of the immediately preceding lines (keep "
                      "pronouns/register/terminology consistent; do NOT retranslate)",
    instruction='Translate the following lines (each line\'s "src" text, within its '
                '"budget" length units). Return ONLY a JSON array like '
                '[{"i": <n>, "text": "<translation>"}] with exactly the given i indices.',
)


VIETNAMESE = LanguageProfile(
    locale="vi", iso639_2="vie", script="Latn", text_direction="ltr",
    length_model="syllable", rate=4.3, cps_max=21.0, expansion_factor=1.0,
    counter=vi_syllable_count, chunk_budget=60,
    system_prompt=_VI_SYSTEM, src_lang_label="tiếng Anh", tgt_lang_label="tiếng Việt",
    english_name="Vietnamese", output_key="vi", context_labels=_VI_LABELS,
    tts_engine="soniox", voice_strategy="preset", preset_pool=_SONIOX_POOL,
    tts_language_code="vi",
    font="Arial", glyph_sample="Đường", segmentation_rule="whitespace",
)

ENGLISH = LanguageProfile(
    locale="en", iso639_2="eng", script="Latn", text_direction="ltr",
    length_model="cps", rate=15.0, cps_max=17.0, expansion_factor=1.0,
    counter=char_count, chunk_budget=240,
    system_prompt=_EN_SYSTEM, src_lang_label="the source language",
    tgt_lang_label="English", english_name="English", output_key="text",
    context_labels=_EN_LABELS,
    tts_engine="soniox", voice_strategy="preset", preset_pool=_SONIOX_POOL,
    tts_language_code="en",
    font="Arial", glyph_sample="English", segmentation_rule="whitespace",
)

FRENCH = LanguageProfile(
    locale="fr", iso639_2="fra", script="Latn", text_direction="ltr",
    length_model="cps", rate=17.0, cps_max=17.0, expansion_factor=1.18,
    counter=char_count, chunk_budget=240,
    system_prompt=_FR_SYSTEM, src_lang_label="anglais", tgt_lang_label="français",
    english_name="French", output_key="text", context_labels=_EN_LABELS,
    tts_engine="soniox", voice_strategy="preset", preset_pool=_SONIOX_POOL,
    tts_language_code="fr",
    font="Arial", glyph_sample="Françaisçàâêîôû", segmentation_rule="whitespace",
)

SPANISH = LanguageProfile(
    locale="es", iso639_2="spa", script="Latn", text_direction="ltr",
    length_model="cps", rate=17.0, cps_max=17.0, expansion_factor=1.25,
    counter=char_count, chunk_budget=240,
    system_prompt=_ES_SYSTEM, src_lang_label="inglés", tgt_lang_label="español",
    english_name="Spanish", output_key="text", context_labels=_EN_LABELS,
    tts_engine="soniox", voice_strategy="preset", preset_pool=_SONIOX_POOL,
    tts_language_code="es",
    font="Arial", glyph_sample="Españolñáéíóú¿¡", segmentation_rule="whitespace",
)

# Best-effort fallback for any unconfigured BCP-47 tag (gated behind
# --allow-fallback in U4). Latin CPS seed; preset voice strategy so an unvalidated
# cross-lingual clone is never the silent default (KTD7).
GENERIC = LanguageProfile(
    locale="und", iso639_2="und", script="Latn", text_direction="ltr",
    length_model="cps", rate=16.0, cps_max=17.0, expansion_factor=1.15,
    counter=char_count, chunk_budget=240,
    system_prompt=_GENERIC_SYSTEM, src_lang_label="the source language",
    tgt_lang_label="the target language", english_name="the target language",
    output_key="text", context_labels=_EN_LABELS,
    tts_engine="soniox", voice_strategy="preset", preset_pool=_SONIOX_POOL,
    tts_language_code="",
    font="Arial", glyph_sample="Latin", segmentation_rule="whitespace",
)
