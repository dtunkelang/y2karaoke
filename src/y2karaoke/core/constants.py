TITLE_CLEANUP_PATTERNS = [
    r'\s*[|｜]\s*.*$',  # Remove after | or ｜
    r'\s*[\(\[]?\s*(ft\.?|feat\.?|featuring).*?[\)\]]?\s*$',  # Featuring
    r'\s*[\(\[].*?[\)\]]\s*',  # Parentheses/brackets
]

YOUTUBE_SUFFIXES = [
    ' Lyrics', ' Official Video', ' Official Audio',
    ' Official Music Video', ' Audio', ' Video'
]

DESCRIPTION_PATTERNS = [
    'is the first track', 'is the second track', 'is the third track',
    'is a song by', 'was released as', 'Read More', 'studio album',
    'music video featuring'
]
DESCRIPTION_REGEX = re.compile("|".join(map(re.escape, DESCRIPTION_PATTERNS)), re.IGNORECASE)

TRANSLATION_LANGUAGES = ['Türkçe','Français','Español','Deutsch','Português']
