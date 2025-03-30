MIN_SCORE = 12

HIGH_SCORE = 15
MEDIUM_SCORE = 5
LOW_SCORE = 2

HIGH_AI_WORDS = {
    "ai", "ml", "intelligence", "deeplearning",
    "artificialintelligence",
    "deeplearning", "neuralnetworks",
    "machinelearning",
    "naturallanguageprocessing", "nlp",
    "largelanguagemodel", "llm",

    # 🔷 OpenAI
    "chatgpt",
    "gpt4", "gpt.4",
    "gpt3.5", "gpt.3.5",
    "openai", "dalle", "dall.e",
    "sora", "whisper",

    # 🔷 Anthropic
    "claude", "claude2", "claude.2",
    "claude3", "claude.3",
    "anthropic",

    # 🔷 Google / DeepMind
    "gemini", "gemini1.5", "gemini.1.5",
    "bard", "palm", "palm2", "palm.2",
    "deepmind", "gemma",

    # 🔷 Meta
    "llama", "llama2", "llama.2",
    "llama3", "llama.3",

    # 🔷 Mistral
    "mistral", "mixtral",

    # 🔷 Cohere
    "commandr", "commandr+",

    # 🔷 xAI
    "grok",

    # 🔷 China AI (Baidu, Alibaba, etc.)
    "ernie", "wenxin", "qwen", "tongyi",
    "pangu", "pan.gu",
    "yayi", "hua", "skywork",

    "deepseek", "deep.seek",
    "deepseekvl", "deepseek.vl",
    "deepseekcoder", "deepseek.coder",
    "deepseekmoe", "deepseek.moe"

    # 🔷 Microsoft
                   "copilot", "phi", "phi2", "phi.2",

    # 🔷 Open Source / Community
    "vicuna", "alpaca", "guanaco",
    "falcon", "stablelm", "replit",
    "notus", "orca", "zephyr", "mpt", "codellama", "code.llama"
}
MEDIUM_AI_WORDS = {
    "data", "algorithm", "crawler", "generation", "automated", "automation", "analysis",
    "sentiment", "processing", "artificial", "predictive"
}
LOW_AI_WORDS = {
    "chatbot", "bot", "machine", "assistant", "prompt"
}

AI_DICT_DETECTION = {word.lower(): HIGH_SCORE for word in HIGH_AI_WORDS}
AI_DICT_DETECTION.update({word.lower(): MEDIUM_SCORE for word in MEDIUM_AI_WORDS})
AI_DICT_DETECTION.update({word.lower(): LOW_SCORE for word in LOW_AI_WORDS})

HIGH_SAAS_WORDS = {
    "dashboard", "subscription", "cloud-based", "platform", "software as a service", "web app", "multi-tenant", "saas",
    "login", "log in", "sign up", "pricing", "api", "register", "account", "demo", "online software", "free trial",
}
MEDIUM_SAAS_WORDS = {
     "users", "plans", "admin panel",
}
LOW_SAAS_WORDS = {
     "get started", "access anywhere", "no downloads", "scale",
}

SAAS_DICT_DETECTION = {word.lower(): HIGH_SCORE for word in HIGH_SAAS_WORDS}
SAAS_DICT_DETECTION.update({word.lower(): MEDIUM_SCORE for word in MEDIUM_SAAS_WORDS})
SAAS_DICT_DETECTION.update({word.lower(): LOW_SCORE for word in LOW_SAAS_WORDS})

EXTERNAL_SOURCES = [
    "https://www.linkedin.com/search/results/companies/?keywords={query}",
    "https://en.wikipedia.org/wiki/{query}",
    "https://twitter.com/search?q={query}&src=typed_query"
]