"""UI strings for Neurox (Turkish / English).

Usage::

    from ui.i18n import t, set_language
    set_language(settings.language)
    st.button(t("clear_chat"))
    t("approval_title", tool="run_python")   # str.format placeholders
"""
from __future__ import annotations

LANGUAGES = {"tr": "🇹🇷 Türkçe", "en": "🇬🇧 English"}
DEFAULT_LANGUAGE = "tr"

STRINGS: dict[str, dict[str, str]] = {
    "tr": {
        # general
        "app_title": "Neurox",
        "app_subtitle": "Yerel yapay zekâ asistanın · {provider} · {model}",
        "welcome": "Merhaba! Ben **Neurox**. Bir şey sor, dosya ya da görsel ekle; araçlarımla senin için iş yapabilirim.",
        "chat_placeholder": "Mesajınızı yazın veya dosya ekleyin...",
        "chat_disabled_placeholder": "Önce bekleyen işlemi onaylayın veya reddedin...",
        "init_error": "Neurox başlatılamadı: {error}",
        # sidebar: cortex
        "sidebar_title": "🧠 Neurox Cortex",
        "section_connection": "🔌 Bağlantı",
        "provider": "Sağlayıcı",
        "provider_ollama": "Ollama",
        "provider_lmstudio": "LM Studio",
        "provider_openai_compat": "OpenAI uyumlu",
        "base_url": "Sunucu adresi (base URL)",
        "api_key": "API anahtarı (gerekirse)",
        "connected": "✅ Bağlı · {count} model",
        "not_connected": "❌ Sunucuya ulaşılamıyor",
        "hint_ollama": "Ollama'yı kurun (ollama.com) ve `ollama serve` çalıştırın, ardından `ollama pull llama3.2` ile bir model indirin.",
        "hint_lmstudio": "LM Studio → **Developer** sekmesi → **Start Server** (port 1234) ve bir model yükleyin.",
        "hint_openai_compat": "OpenAI uyumlu sunucunuzu (llama.cpp `llama-server`, vLLM...) başlatın ve adresi kontrol edin.",
        "refresh": "🔄 Yenile",
        "error_details": "ℹ️ Hata ayrıntısı",
        "section_models": "🤖 Modeller",
        "chat_model": "Sohbet modeli",
        "chat_model_free": "Sohbet modeli (elle yazın)",
        "model_missing": "⚠️ `{model}` sunucuda bulunamadı.",
        "no_models": "Sunucuda model yok. Önce bir model indirin/yükleyin.",
        "embed_model": "Embedding modeli",
        "temperature": "Sıcaklık (yaratıcılık)",
        "section_tools": "🛠️ Araçlar",
        "tools_enabled": "Araç kullanımı (tool calling)",
        "enabled_tools": "Etkin araçlar",
        "computer_use": "🖱️ Bilgisayar kontrolü",
        "computer_use_warning": "Neurox fareni ve klavyeni kontrol edebilir, ekran görüntüsü alabilir. Yalnızca güvendiğin modellerle aç; acil durumda fareyi ekranın bir köşesine götür.",
        "computer_available": "✅ Bilgisayar kontrolü kullanılabilir",
        "computer_unavailable": "❌ Kullanılamıyor: {reason}",
        "auto_approve": "Tehlikeli araçları otomatik onayla",
        "auto_approve_warning": "Kod çalıştırma, dosya yazma ve bilgisayar kontrolü SORMADAN yapılacak. Önerilmez.",
        "max_agent_steps": "En fazla ajan adımı",
        "section_display": "🎛️ Görünüm",
        "show_thoughts": "İç sesi göster 🧠",
        "language": "Dil",
        "settings_saved": "Ayarlar kaydedildi",
        # sidebar: expanders
        "teach_title": "🎓 Neurox'a öğret",
        "teach_help": "Bir hatayı düzelt veya yeni bilgi ver; Neurox bunu hafızasına kaydeder.",
        "teach_submit": "Düzeltmeyi gönder",
        "teach_saved": "Düzeltme hafızaya kaydedildi! 🧠",
        "teach_empty": "Lütfen bir metin girin.",
        "teach_error": "Kaydedilemedi: {error}",
        "rules_title": "📜 Kurallar",
        "rules_empty": "Henüz kural yok.",
        "rules_add": "Yeni kural",
        "rules_add_button": "➕ Ekle",
        "rules_unavailable": "Kural modülü kullanılamıyor.",
        "delete": "Sil",
        "memory_title": "🧠 Hafıza",
        "memory_count": "Kayıtlı anı sayısı: **{count}**",
        "memory_unavailable": "Hafıza modülü kullanılamıyor.",
        "memory_confirm": "Eminim, tüm hafızayı sil",
        "memory_clear": "🗑️ Hafızayı temizle",
        "memory_cleared": "Hafıza temizlendi.",
        "xp_title": "🏆 Seviye",
        "xp_label": "Seviye {level} · {xp} XP",
        "xp_unavailable": "XP bilgisi yok.",
        "clear_chat": "🧹 Sohbeti temizle",
        # turn
        "thinking": "Düşünüyor...",
        "done": "Tamamlandı",
        "turn_error": "Bir hata oluştu",
        "waiting_approval": "Onay bekleniyor",
        "steps": "Adımlar",
        "inner_monologue": "🧠 İç ses",
        "error_reply": "⚠️ Yanıt üretilemedi: {error}",
        "empty_reply": "_(boş yanıt)_",
        "ev_tool_call": "🔧 **{name}** çağrılıyor",
        "ev_tool_result": "📤 **{name}** sonucu",
        "ev_approval": "⏸️ **{name}** için onay gerekiyor",
        "ev_denied": "🚫 Reddedildi",
        "ev_error": "Hata: {text}",
        "attachments": "📎 Ekler",
        "attachment_truncated": "[... kısaltıldı]",
        "attachment_unsupported": "Desteklenmeyen dosya atlandı: {name}",
        # approval
        "approval_title": "Neurox **{tool}** aracını çalıştırmak istiyor",
        "approval_help": "Bu araç bilgisayarında değişiklik yapabilir. Argümanları kontrol et.",
        "approve": "✅ Onayla",
        "deny": "❌ Reddet",
        "approval_lost": "Bekleyen işlem bulunamadı.",
        "agent_missing": "Ajan modülü kullanılamıyor: {error}",
    },
    "en": {
        "app_title": "Neurox",
        "app_subtitle": "Your local AI assistant · {provider} · {model}",
        "welcome": "Hi! I'm **Neurox**. Ask me anything, attach files or images; I can use my tools to get things done.",
        "chat_placeholder": "Type a message or attach files...",
        "chat_disabled_placeholder": "Approve or deny the pending action first...",
        "init_error": "Could not start Neurox: {error}",
        "sidebar_title": "🧠 Neurox Cortex",
        "section_connection": "🔌 Connection",
        "provider": "Provider",
        "provider_ollama": "Ollama",
        "provider_lmstudio": "LM Studio",
        "provider_openai_compat": "OpenAI-compatible",
        "base_url": "Server address (base URL)",
        "api_key": "API key (if required)",
        "connected": "✅ Connected · {count} models",
        "not_connected": "❌ Server not reachable",
        "hint_ollama": "Install Ollama (ollama.com) and run `ollama serve`, then pull a model with `ollama pull llama3.2`.",
        "hint_lmstudio": "LM Studio → **Developer** tab → **Start Server** (port 1234) and load a model.",
        "hint_openai_compat": "Start your OpenAI-compatible server (llama.cpp `llama-server`, vLLM...) and check the URL.",
        "refresh": "🔄 Refresh",
        "error_details": "ℹ️ Error details",
        "section_models": "🤖 Models",
        "chat_model": "Chat model",
        "chat_model_free": "Chat model (type it)",
        "model_missing": "⚠️ `{model}` is not available on the server.",
        "no_models": "No models on the server. Pull/load a model first.",
        "embed_model": "Embedding model",
        "temperature": "Temperature (creativity)",
        "section_tools": "🛠️ Tools",
        "tools_enabled": "Tool use (tool calling)",
        "enabled_tools": "Enabled tools",
        "computer_use": "🖱️ Computer use",
        "computer_use_warning": "Neurox can control your mouse and keyboard and take screenshots. Only enable with models you trust; move the mouse to a screen corner to abort.",
        "computer_available": "✅ Computer use available",
        "computer_unavailable": "❌ Unavailable: {reason}",
        "auto_approve": "Auto-approve dangerous tools",
        "auto_approve_warning": "Code execution, file writes and computer control will run WITHOUT asking. Not recommended.",
        "max_agent_steps": "Max agent steps",
        "section_display": "🎛️ Display",
        "show_thoughts": "Show inner monologue 🧠",
        "language": "Language",
        "settings_saved": "Settings saved",
        "teach_title": "🎓 Teach Neurox",
        "teach_help": "Correct a mistake or teach a fact; Neurox stores it in memory.",
        "teach_submit": "Submit correction",
        "teach_saved": "Correction saved to memory! 🧠",
        "teach_empty": "Please enter some text.",
        "teach_error": "Could not save: {error}",
        "rules_title": "📜 Rules",
        "rules_empty": "No rules yet.",
        "rules_add": "New rule",
        "rules_add_button": "➕ Add",
        "rules_unavailable": "Rules module unavailable.",
        "delete": "Delete",
        "memory_title": "🧠 Memory",
        "memory_count": "Stored memories: **{count}**",
        "memory_unavailable": "Memory module unavailable.",
        "memory_confirm": "I'm sure, wipe all memory",
        "memory_clear": "🗑️ Clear memory",
        "memory_cleared": "Memory cleared.",
        "xp_title": "🏆 Level",
        "xp_label": "Level {level} · {xp} XP",
        "xp_unavailable": "No XP data.",
        "clear_chat": "🧹 Clear chat",
        "thinking": "Thinking...",
        "done": "Done",
        "turn_error": "Something went wrong",
        "waiting_approval": "Waiting for approval",
        "steps": "Steps",
        "inner_monologue": "🧠 Inner monologue",
        "error_reply": "⚠️ Could not produce a reply: {error}",
        "empty_reply": "_(empty reply)_",
        "ev_tool_call": "🔧 Calling **{name}**",
        "ev_tool_result": "📤 **{name}** result",
        "ev_approval": "⏸️ **{name}** needs approval",
        "ev_denied": "🚫 Denied",
        "ev_error": "Error: {text}",
        "attachments": "📎 Attachments",
        "attachment_truncated": "[... truncated]",
        "attachment_unsupported": "Skipped unsupported file: {name}",
        "approval_title": "Neurox wants to run **{tool}**",
        "approval_help": "This tool can change things on your computer. Check the arguments.",
        "approve": "✅ Approve",
        "deny": "❌ Deny",
        "approval_lost": "No pending action found.",
        "agent_missing": "Agent module unavailable: {error}",
    },
}

# Strings for the redesigned layout (welcome screen, suggestion cards, sidebar groups).
STRINGS["tr"].update({
    "new_chat": "＋  Yeni sohbet",
    "welcome_title": "Bugün sana nasıl yardım edebilirim?",
    "welcome_sub": "Soru sor, dosya veya görsel ekle. Web'de arayabilir, kod çalıştırabilir, hatta bilgisayarını kullanabilirim.",
    "sugg1_title": "🔎 Web'de araştır", "sugg1_prompt": "Yapay zekâ dünyasındaki bu haftanın en önemli haberlerini araştır ve özetle.",
    "sugg2_title": "🐍 Kod yaz ve çalıştır", "sugg2_prompt": "Python ile 1'den 100'e kadar asal sayıları bulan bir kod yaz ve çalıştır.",
    "sugg3_title": "🧠 Bir konuyu öğret", "sugg3_prompt": "Bana adım adım makine öğrenmesinin temellerini öğret.",
    "sugg4_title": "📋 Plan çıkar", "sugg4_prompt": "Bir mobil uygulama projesi için 4 haftalık bir plan hazırla ve riskleri değerlendir.",
    "section_computer": "🖱️ Bilgisayar kontrolü",
    "section_learning": "🎓 Hafıza ve öğrenme",
    "status_online": "Çevrimiçi",
    "status_offline": "Çevrimdışı",
    "level": "Seviye",
    "footer": "Neurox yerel çalışır · verilerin bilgisayarında kalır",
    "friendly_unreachable": "⚠️ **Model sunucusuna bağlanamadım.**\n\n"
                            "Ollama'yı başlat (`ollama serve`) ya da LM Studio'da yerel sunucuyu aç, sonra tekrar dene. "
                            "Bağlantı ayarları sol menüde **🔌 Bağlantı** bölümünde.",
})
STRINGS["en"].update({
    "new_chat": "＋  New chat",
    "welcome_title": "How can I help you today?",
    "welcome_sub": "Ask anything, attach files or images. I can search the web, run code and even use your computer.",
    "sugg1_title": "🔎 Research the web", "sugg1_prompt": "Research the most important AI news of this week and summarize it.",
    "sugg2_title": "🐍 Write & run code", "sugg2_prompt": "Write and run Python code that finds all prime numbers from 1 to 100.",
    "sugg3_title": "🧠 Teach me something", "sugg3_prompt": "Teach me the basics of machine learning step by step.",
    "sugg4_title": "📋 Make a plan", "sugg4_prompt": "Create a 4-week plan for a mobile app project and assess the risks.",
    "section_computer": "🖱️ Computer use",
    "section_learning": "🎓 Memory & learning",
    "status_online": "Online",
    "status_offline": "Offline",
    "level": "Level",
    "footer": "Neurox runs locally · your data stays on your computer",
    "friendly_unreachable": "⚠️ **I couldn't reach the model server.**\n\n"
                            "Start Ollama (`ollama serve`) or LM Studio's local server, then try again. "
                            "Connection settings are under **🔌 Connection** in the sidebar.",
})

_current = DEFAULT_LANGUAGE


def set_language(lang: str | None) -> None:
    global _current
    _current = lang if lang in STRINGS else DEFAULT_LANGUAGE


def get_language() -> str:
    return _current


def t(key: str, lang: str | None = None, **fmt) -> str:
    """Translate ``key`` into the current (or given) language; falls back to English, then the key itself."""
    table = STRINGS.get(lang or _current, STRINGS[DEFAULT_LANGUAGE])
    text = table.get(key) or STRINGS["en"].get(key) or key
    if fmt:
        try:
            return text.format(**fmt)
        except (KeyError, IndexError, ValueError):
            return text
    return text
