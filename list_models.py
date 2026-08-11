"""
Script to list available Gemini models using Google's GenAI SDKs.
"""
import os
import sys
from dotenv import load_dotenv

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

def main():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    if len(sys.argv) > 1 and sys.argv[1]:
        api_key = sys.argv[1]

    if not api_key:
        print("⚠️ No API Key found in environment or .env file.")
        print("💡 Get your free API key at: https://aistudio.google.com/app/apikey")
        print("Usage: python list_models.py <YOUR_API_KEY>")
        return

    print("🔍 Checking available Gemini models...\n")
    
    success = False
    
    # 1. Try modern google-genai SDK
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        models = list(client.models.list())
        print("✅ Models available via `google-genai`:")
        for m in models:
            name = getattr(m, "name", str(m))
            display_name = getattr(m, "display_name", "")
            print(f"  • {name} {f'({display_name})' if display_name else ''}")
        success = True
    except Exception as e:
        print(f"ℹ️ google-genai SDK notice: {e}")

    # 2. Try google-generativeai SDK as fallback
    if not success:
        try:
            import google.generativeai as genai_legacy
            genai_legacy.configure(api_key=api_key)
            models = genai_legacy.list_models()
            print("\n✅ Models available via `google-generativeai`:")
            for m in models:
                if hasattr(m, 'supported_generation_methods') and 'generateContent' in m.supported_generation_methods:
                    print(f"  • {m.name} ({m.display_name})")
            success = True
        except Exception as e:
            print(f"❌ Error listing models via google-generativeai: {e}")

if __name__ == "__main__":
    main()
