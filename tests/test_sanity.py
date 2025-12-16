from src.config import get_settings


def test_settings_defaults():
    settings = get_settings()
    assert "en" in settings.allowed_languages
