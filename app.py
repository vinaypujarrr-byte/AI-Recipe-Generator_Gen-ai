import os
import sys
import json
import re
import time
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="ChefAI - Intelligent Gourmet Recipe Generator",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    /* Global Styles & Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Main Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
    }
    
    /* Custom Card Containers */
    .recipe-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
    }
    
    .hero-container {
        background: linear-gradient(135deg, rgba(234, 88, 12, 0.2) 0%, rgba(168, 85, 247, 0.2) 100%);
        border: 1px solid rgba(251, 146, 60, 0.3);
        border-radius: 20px;
        padding: 32px;
        text-align: center;
        margin-bottom: 30px;
    }
    
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #fb923c, #f43f5e, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        max-width: 650px;
        margin: 0 auto;
    }

    /* Badge Pills */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .badge-orange { background: rgba(249, 115, 22, 0.2); color: #fb923c; border: 1px solid rgba(249, 115, 22, 0.4); }
    .badge-purple { background: rgba(168, 85, 247, 0.2); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.4); }
    .badge-green  { background: rgba(34, 197, 94, 0.2);  color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.4); }
    .badge-blue   { background: rgba(59, 130, 246, 0.2);  color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); }
    
    /* Section Headers */
    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #f1f5f9;
        border-bottom: 2px solid rgba(251, 146, 60, 0.4);
        padding-bottom: 8px;
        margin-top: 18px;
        margin-bottom: 16px;
    }
    
    /* Nutrition Stats Box */
    .stat-box {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 12px;
        text-align: center;
    }
    .stat-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #fb923c;
    }
    .stat-label {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Customize Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #ea580c, #d97706);
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 12px;
        padding: 10px 24px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px 0 rgba(234, 88, 12, 0.39);
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #f97316, #f59e0b);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(234, 88, 12, 0.55);
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# GEMINI API HELPER FUNCTIONS
# ==========================================

def get_gemini_client(api_key: str):
    """
    Initialize Gemini client. Tries google.genai first, falls back to google.generativeai.
    """
    if not api_key:
        return None, "NO_KEY", "API Key is missing."
    
    # Try official google.genai
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        return client, "GENAI_SDK", None
    except Exception as e:
        # Fallback to google.generativeai
        try:
            import google.generativeai as genai_legacy
            genai_legacy.configure(api_key=api_key)
            return genai_legacy, "LEGACY_SDK", None
        except Exception as e2:
            return None, "ERROR", f"Failed to initialize SDK: {e2}"


def get_available_models(client, sdk_type: str) -> list:
    """
    Dynamically query all available generateContent models for this API key.
    """
    valid_models = []
    if sdk_type == "GENAI_SDK":
        try:
            models_pager = client.models.list()
            for m in models_pager:
                name = getattr(m, "name", str(m))
                name = name.replace("models/", "")
                valid_models.append(name)
        except Exception:
            pass
    elif sdk_type == "LEGACY_SDK":
        try:
            for m in client.list_models():
                methods = getattr(m, 'supported_generation_methods', [])
                if 'generateContent' in methods:
                    name = m.name.replace("models/", "")
                    valid_models.append(name)
        except Exception:
            pass

    default_fallbacks = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro-latest",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-2.0-flash-lite"
    ]

    combined = [m for m in valid_models if m] + default_fallbacks
    return list(dict.fromkeys(combined))


def generate_gemini_response(client, sdk_type: str, model_name: str, prompt: str, image=None, system_instruction: str = None, temperature: float = 0.7):
    """
    Call Gemini API supporting both google.genai and google.generativeai SDKs
    with dynamic model resolution, fallback retries, and multimodal image input.
    """
    clean_requested = model_name.replace("models/", "") if model_name != "Auto (Best Available)" else ""
    available_models = get_available_models(client, sdk_type)

    if clean_requested:
        models_to_try = [clean_requested] + available_models
    else:
        models_to_try = available_models

    models_to_try = list(dict.fromkeys(models_to_try))
    last_error = None

    if sdk_type == "GENAI_SDK":
        from google.genai import types
        contents_payload = [image, prompt] if image is not None else prompt

        for m in models_to_try:
            # 1. Try with response_mime_type="application/json"
            try:
                config = types.GenerateContentConfig(
                    temperature=temperature,
                    response_mime_type="application/json"
                )
                if system_instruction:
                    config.system_instruction = system_instruction
                
                response = client.models.generate_content(
                    model=m,
                    contents=contents_payload,
                    config=config
                )
                if response.text:
                    return response.text
            except Exception as err:
                last_error = err
                # 2. Try without response_mime_type if json mode failed
                try:
                    sys_inst = (system_instruction or "") + "\nRespond STRICTLY in valid JSON format."
                    config_plain = types.GenerateContentConfig(
                        temperature=temperature,
                        system_instruction=sys_inst
                    )
                    response = client.models.generate_content(
                        model=m,
                        contents=contents_payload,
                        config=config_plain
                    )
                    if response.text:
                        return response.text
                except Exception as err2:
                    last_error = err2
                    continue
        raise last_error

    elif sdk_type == "LEGACY_SDK":
        import google.generativeai as genai_legacy
        contents_payload = [prompt, image] if image is not None else prompt

        for m in models_to_try:
            # 1. Try with response_mime_type JSON
            try:
                gen_model = genai_legacy.GenerativeModel(
                    model_name=m,
                    system_instruction=system_instruction,
                    generation_config={"temperature": temperature, "response_mime_type": "application/json"}
                )
                response = gen_model.generate_content(contents_payload)
                if response.text:
                    return response.text
            except Exception as err:
                last_error = err
                # 2. Try plain text generation asking for JSON
                try:
                    sys_inst = (system_instruction or "") + "\nIMPORTANT: Return ONLY raw valid JSON."
                    gen_model = genai_legacy.GenerativeModel(
                        model_name=m,
                        system_instruction=sys_inst,
                        generation_config={"temperature": temperature}
                    )
                    response = gen_model.generate_content(contents_payload)
                    if response.text:
                        return response.text
                except Exception as err2:
                    last_error = err2
                    continue
        raise last_error
    else:
        raise ValueError("Invalid SDK type or client not initialized.")


def parse_json_response(raw_text: str):
    """
    Safely parse JSON response from Gemini, handling markdown fence blocks if present.
    """
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        # Remove code blocks like ```json ... ```
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
        cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Match first '{' or '[' to last '}' or ']'
        match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError("Could not extract valid JSON from response.")


def normalize_recipe_data(data) -> dict:
    """
    Ensure recipe_data is always a dictionary.
    """
    if isinstance(data, list) and len(data) > 0:
        data = data[0]
    if not isinstance(data, dict):
        data = {}
    return data


def extract_meal_info(meal_obj, default_type="Meal", default_cal=400):
    """
    Safely extract title, brief description, and calories from any dict or string object.
    """
    if isinstance(meal_obj, str) and meal_obj.strip():
        return {"title": meal_obj.strip(), "brief": "Custom recommended meal", "calories": default_cal}
    
    if not isinstance(meal_obj, dict):
        return {"title": f"{default_type} Choice", "brief": "Nutritious balanced option", "calories": default_cal}
    
    # Extract title from potential keys
    title = (
        meal_obj.get("title") or 
        meal_obj.get("name") or 
        meal_obj.get("recipe") or 
        meal_obj.get("dish") or 
        meal_obj.get("item") or 
        meal_obj.get("meal") or 
        f"{default_type} Choice"
    )
    
    # Extract brief/description from potential keys
    brief = (
        meal_obj.get("brief") or 
        meal_obj.get("description") or 
        meal_obj.get("details") or 
        meal_obj.get("instructions") or 
        meal_obj.get("summary") or 
        "Nutritious chef recommendation"
    )
    
    # Extract calories from potential keys
    raw_cal = (
        meal_obj.get("calories") or 
        meal_obj.get("kcal") or 
        meal_obj.get("cal") or 
        meal_obj.get("energy") or 
        default_cal
    )
    try:
        calories = int(raw_cal)
        if calories <= 0:
            calories = default_cal
    except (ValueError, TypeError):
        calories = default_cal
        
    return {"title": str(title), "brief": str(brief), "calories": calories}


def normalize_meal_plan(plan_data) -> dict:
    """
    Ensure meal_plan is always a dictionary with expected plan_title, daily_summary, and complete days list.
    """
    title = "Custom Multi-Day Meal Plan"
    summary = "Nutritional meal plan tailored to your preferences."
    raw_days = []

    if isinstance(plan_data, list):
        raw_days = plan_data
    elif isinstance(plan_data, dict):
        title = plan_data.get("plan_title") or plan_data.get("title") or title
        summary = plan_data.get("daily_summary") or plan_data.get("summary") or summary
        
        if "days" in plan_data and isinstance(plan_data["days"], list):
            raw_days = plan_data["days"]
        else:
            # Check for keys like "day_1", "day1", "Day 1", etc.
            for k, v in plan_data.items():
                if isinstance(v, dict) and any(w in str(k).lower() for w in ["day", "daily"]):
                    raw_days.append(v)
            if not raw_days:
                # If values are dicts containing meal info
                for k, v in plan_data.items():
                    if isinstance(v, dict) and any(m in v for m in ["breakfast", "lunch", "dinner", "snack", "meals"]):
                        raw_days.append(v)

    normalized_days = []
    for idx, d in enumerate(raw_days):
        if not isinstance(d, dict):
            continue
        
        day_num = d.get("day_number") or d.get("day") or (idx + 1)
        day_name = d.get("day_name") or d.get("name") or f"Day {day_num}"
        
        # Handle 'meals' array if Gemini returned meals as a list instead of key-value pairs
        if "meals" in d and isinstance(d["meals"], list):
            meals_arr = d["meals"]
            b_obj = meals_arr[0] if len(meals_arr) > 0 else {}
            l_obj = meals_arr[1] if len(meals_arr) > 1 else {}
            din_obj = meals_arr[2] if len(meals_arr) > 2 else {}
            s_obj = meals_arr[3] if len(meals_arr) > 3 else {}
        else:
            b_obj = d.get("breakfast") or d.get("Breakfast") or {}
            l_obj = d.get("lunch") or d.get("Lunch") or {}
            din_obj = d.get("dinner") or d.get("Dinner") or {}
            s_obj = d.get("snack") or d.get("Snack") or {}
            
        normalized_days.append({
            "day_number": day_num,
            "day_name": day_name,
            "breakfast": extract_meal_info(b_obj, "Breakfast", 400),
            "lunch": extract_meal_info(l_obj, "Lunch", 600),
            "dinner": extract_meal_info(din_obj, "Dinner", 700),
            "snack": extract_meal_info(s_obj, "Snack", 300)
        })

    return {
        "plan_title": title,
        "daily_summary": summary,
        "days": normalized_days
    }


def calculate_estimated_cost(recipe: dict) -> dict:
    """
    Calculate estimated grocery cost and per-serving price.
    """
    ingredients = recipe.get("ingredients", [])
    servings = recipe.get("servings", 2)
    try:
        servings = int(servings)
        if servings <= 0:
            servings = 2
    except (ValueError, TypeError):
        servings = 2

    item_count = len(ingredients)
    total_est = round(max(item_count, 1) * 1.85 + 2.00, 2)
    per_serv = round(total_est / servings, 2)

    tier = "$"
    if per_serv > 5.50:
        tier = "$$$ (Gourmet / Premium)"
    elif per_serv > 3.20:
        tier = "$$ (Moderate)"
    else:
        tier = "$ (Budget-Friendly)"

    return {
        "total_cost": f"${total_est:.2f}",
        "per_serving": f"${per_serv:.2f}",
        "tier": tier
    }


# ==========================================
# SIDEBAR CONTROLS
# ==========================================

st.sidebar.image("https://img.icons8.com/emoji/96/000000/cooking-pot.png", width=70)
st.sidebar.title("👨‍🍳 ChefAI Settings")

# API Key handling
env_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
user_api_key = st.sidebar.text_input(
    "🔑 Gemini API Key",
    value=env_api_key,
    type="password",
    help="Enter your Gemini API key. Get one for free at https://aistudio.google.com/app/apikey"
)

if not user_api_key:
    st.sidebar.warning("⚠️ API Key required. Get yours at [Google AI Studio](https://aistudio.google.com/app/apikey).")

# Model selection
selected_model = st.sidebar.selectbox(
    "🤖 Gemini Model",
    options=["Auto (Best Available)", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash-latest", "gemini-1.5-pro-latest", "gemini-1.5-flash", "gemini-1.5-pro"],
    index=0,
    help="Select the AI model for generation. 'Auto' automatically selects the best available model for your API key."
)

creativity = st.sidebar.slider(
    "🎨 Recipe Creativity",
    min_value=0.1,
    max_value=1.0,
    value=0.7,
    step=0.05,
    help="Higher values produce more exotic/fusion recipes."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Quick Tips")
st.sidebar.info(
    "• Enter all ingredients you currently have in your kitchen.\n"
    "• Add specific dietary preferences like Keto, Vegan, or Gluten-Free.\n"
    "• Use the Meal Planner tab for weekly meal prep!"
)


# ==========================================
# HERO HEADER
# ==========================================

st.markdown("""
<div class="hero-container">
    <div class="hero-title">🍳 AI Gourmet Recipe & Meal Generator</div>
    <div class="hero-subtitle">Transform your available ingredients into chef-crafted recipes, detailed nutritional breakdowns, and personalized weekly meal plans instantly.</div>
</div>
""", unsafe_allow_html=True)


# ==========================================
# MAIN TAB ARCHITECTURE
# ==========================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🥘 Instant Recipe Generator",
    "📸 Fridge & Photo Scanner",
    "👨‍🍳 Hands-Free Cooking Studio",
    "🗓️ Weekly Meal Planner",
    "🔄 Recipe Customizer & Transformer",
    "🛒 Shopping List Hub"
])


# ------------------------------------------
# TAB 1: INSTANT RECIPE GENERATOR
# ------------------------------------------
with tab1:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### 🛒 Available Ingredients")
        ingredients_input = st.text_area(
            "List ingredients (comma-separated):",
            value="chicken breast, garlic, olive oil, spinach, parmesan cheese, cherry tomatoes",
            height=100,
            placeholder="e.g. eggs, potatoes, cheddar cheese, onion, spinach..."
        )

        include_pantry = st.checkbox(
            "🧂 Include common pantry staples (Salt, Pepper, Water, Cooking Oil)",
            value=True
        )

        col_a, col_b = st.columns(2)
        with col_a:
            cuisine_type = st.selectbox(
                "🌍 Cuisine Style",
                options=["Any / Fusion", "Italian", "Indian", "Mexican", "Japanese", "Mediterranean", "Thai", "French", "American", "Chinese"]
            )
            cooking_time = st.selectbox(
                "⏱️ Max Cook Time",
                options=["Any Time", "< 15 mins (Quick)", "< 30 mins", "< 45 mins", "< 60 mins"]
            )
        with col_b:
            meal_type = st.selectbox(
                "🍽️ Meal Type",
                options=["Dinner", "Lunch", "Breakfast", "Snack / Appetizer", "Dessert", "Healthy Beverage"]
            )
            skill_level = st.selectbox(
                "🍳 Cooking Skill Level",
                options=["Easy / Beginner", "Intermediate", "Master Chef"]
            )

    with col_right:
        st.markdown("### 🥗 Dietary Restrictions & Preferences")
        dietary_options = st.multiselect(
            "Select all that apply:",
            options=["High Protein", "Vegetarian", "Vegan", "Gluten-Free", "Keto", "Low Carb", "Dairy-Free", "Nut-Free", "Halal"],
            default=["High Protein"]
        )

        servings = st.slider("👥 Number of Servings", min_value=1, max_value=12, value=2)

        special_requests = st.text_input(
            "✨ Special Requests or Notes (Optional)",
            placeholder="e.g., Crispy texture, spicy flavor, kid-friendly..."
        )

        st.markdown("<br>", unsafe_allow_html=True)
        generate_btn = st.button("🚀 Generate Recipe Now", use_container_width=True)

    # Execution when button clicked
    if generate_btn:
        if not user_api_key:
            st.error("🔑 Please enter a valid Gemini API Key in the sidebar to proceed!")
        elif not ingredients_input.strip():
            st.warning("⚠️ Please provide at least one ingredient.")
        else:
            client, sdk_type, err_msg = get_gemini_client(user_api_key)
            if not client:
                st.error(err_msg)
            else:
                with st.spinner("🧑‍🍳 Chef AI is crafting your personalized recipe..."):
                    system_prompt = """
                    You are an expert World-Class Executive Chef and Nutritionist.
                    Create a detailed, delicious recipe based strictly on the user's constraints.
                    
                    Return ONLY a JSON object matching this exact schema:
                    {
                      "recipe_title": "string",
                      "tagline": "string",
                      "prep_time_mins": number,
                      "cook_time_mins": number,
                      "total_time_mins": number,
                      "servings": number,
                      "difficulty": "Easy|Intermediate|Hard",
                      "cuisine": "string",
                      "dietary_tags": ["string"],
                      "nutrition_per_serving": {
                        "calories": number,
                        "protein_g": number,
                        "carbs_g": number,
                        "fat_g": number,
                        "fiber_g": number
                      },
                      "ingredients": [
                        { "item": "string", "amount": "string", "category": "produce|protein|dairy|pantry|spices|other" }
                      ],
                      "instructions": [
                        { "step_number": number, "title": "string", "description": "string" }
                      ],
                      "chef_secret_tips": ["string"],
                      "beverage_pairing": "string"
                    }
                    """

                    user_prompt = f"""
                    Ingredients Available: {ingredients_input}
                    Include Staples: {include_pantry}
                    Cuisine Preference: {cuisine_type}
                    Meal Type: {meal_type}
                    Cook Time Limit: {cooking_time}
                    Skill Level: {skill_level}
                    Dietary Restrictions: {', '.join(dietary_options) if dietary_options else 'None'}
                    Servings: {servings}
                    Special Requests: {special_requests if special_requests else 'None'}
                    """

                    try:
                        raw_response = generate_gemini_response(
                            client=client,
                            sdk_type=sdk_type,
                            model_name=selected_model,
                            prompt=user_prompt,
                            system_instruction=system_prompt,
                            temperature=creativity
                        )
                        
                        recipe_data = normalize_recipe_data(parse_json_response(raw_response))
                        st.session_state["current_recipe"] = recipe_data
                        st.success("🎉 Recipe created successfully!")

                    except Exception as e:
                        st.error(f"❌ Failed to generate recipe: {str(e)}")

    # Display Recipe Output Card if available
    if "current_recipe" in st.session_state:
        recipe = normalize_recipe_data(st.session_state["current_recipe"])

        st.markdown("---")
        st.markdown(f"## 🍽️ {recipe.get('recipe_title', 'Delicious Recipe')}")
        st.markdown(f"*{recipe.get('tagline', '')}*")

        # Badges row
        badge_html = f"<span class='badge badge-orange'>🌍 {recipe.get('cuisine', 'Fusion')}</span>"
        badge_html += f"<span class='badge badge-purple'>⏱️ {recipe.get('total_time_mins', 30)} Mins</span>"
        badge_html += f"<span class='badge badge-blue'>👥 {recipe.get('servings', 2)} Servings</span>"
        badge_html += f"<span class='badge badge-green'>🔥 {recipe.get('difficulty', 'Easy')}</span>"
        for tag in recipe.get("dietary_tags", []):
            badge_html += f"<span class='badge badge-orange'>🥗 {tag}</span>"

        st.markdown(badge_html, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Macro Nutrition Stats
        nutr = recipe.get("nutrition_per_serving", {})
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f"<div class='stat-box'><div class='stat-value'>{nutr.get('calories', 0)}</div><div class='stat-label'>Calories</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='stat-box'><div class='stat-value'>{nutr.get('protein_g', 0)}g</div><div class='stat-label'>Protein</div></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='stat-box'><div class='stat-value'>{nutr.get('carbs_g', 0)}g</div><div class='stat-label'>Carbs</div></div>", unsafe_allow_html=True)
        with c4:
            st.markdown(f"<div class='stat-box'><div class='stat-value'>{nutr.get('fat_g', 0)}g</div><div class='stat-label'>Fat</div></div>", unsafe_allow_html=True)
        with c5:
            st.markdown(f"<div class='stat-box'><div class='stat-value'>{nutr.get('fiber_g', 0)}g</div><div class='stat-label'>Fiber</div></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Ingredients & Cooking Steps Split
        r_col1, r_col2 = st.columns([1, 1.4])

        with r_col1:
            st.markdown("<div class='section-title'>🧺 Ingredients Needed</div>", unsafe_allow_html=True)
            for idx, ing in enumerate(recipe.get("ingredients", [])):
                amt = ing.get("amount", "")
                item = ing.get("item", "")
                st.checkbox(f"**{amt}** {item}", key=f"ing_{idx}")

        with r_col2:
            st.markdown("<div class='section-title'>👨‍🍳 Cooking Instructions</div>", unsafe_allow_html=True)
            for step in recipe.get("instructions", []):
                num = step.get("step_number", 1)
                title = step.get("title", f"Step {num}")
                desc = step.get("description", "")
                st.checkbox(f"**Step {num}: {title}**\n\n{desc}", key=f"step_{num}")

        # Pro Tips & Beverage Pairing
        if recipe.get("chef_secret_tips") or recipe.get("beverage_pairing"):
            st.markdown("---")
            col_tip1, col_tip2 = st.columns(2)
            with col_tip1:
                st.markdown("#### 💡 Chef's Secret Tips")
                for tip in recipe.get("chef_secret_tips", []):
                    st.markdown(f"• {tip}")
            with col_tip2:
                st.markdown("#### 🍷 Suggested Beverage Pairing")
                st.markdown(f"*{recipe.get('beverage_pairing', 'Fresh lemon water or Pinot Noir')}*")

        # Grocery Cost Estimator Box
        cost_info = calculate_estimated_cost(recipe)
        st.markdown("---")
        st.markdown("<div class='section-title'>💰 Grocery Cost Estimator</div>", unsafe_allow_html=True)
        cost_c1, cost_c2, cost_c3 = st.columns(3)
        with cost_c1:
            st.markdown(f"<div class='stat-box'><div class='stat-value'>{cost_info['total_cost']}</div><div class='stat-label'>Est. Total Grocery Cost</div></div>", unsafe_allow_html=True)
        with cost_c2:
            st.markdown(f"<div class='stat-box'><div class='stat-value'>{cost_info['per_serving']}</div><div class='stat-label'>Cost Per Serving</div></div>", unsafe_allow_html=True)
        with cost_c3:
            st.markdown(f"<div class='stat-box'><div class='stat-value'>{cost_info['tier']}</div><div class='stat-label'>Budget Category</div></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Export & PDF Print Options
        st.markdown("---")
        recipe_md = f"# {recipe.get('recipe_title')}\n\n{recipe.get('tagline')}\n\n"
        recipe_md += f"**Prep Time:** {recipe.get('prep_time_mins')}m | **Cook Time:** {recipe.get('cook_time_mins')}m | **Servings:** {recipe.get('servings')}\n"
        recipe_md += f"**Est. Total Cost:** {cost_info['total_cost']} ({cost_info['per_serving']} / serving)\n\n"
        recipe_md += "## Ingredients\n"
        for ing in recipe.get("ingredients", []):
            recipe_md += f"- {ing.get('amount')} {ing.get('item')}\n"
        recipe_md += "\n## Instructions\n"
        for step in recipe.get("instructions", []):
            recipe_md += f"{step.get('step_number')}. **{step.get('title')}**: {step.get('description')}\n"

        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            st.download_button(
                label="📥 Download Recipe as Markdown",
                data=recipe_md,
                file_name=f"{recipe.get('recipe_title', 'recipe').lower().replace(' ', '_')}.md",
                mime="text/markdown",
                use_container_width=True
            )
        with exp_col2:
            st.components.v1.html(
                """
                <button onclick="window.print()" style="
                    background: linear-gradient(90deg, #ea580c, #d97706);
                    color: white; font-weight: bold; border: none; padding: 10px 24px;
                    border-radius: 12px; cursor: pointer; width: 100%; font-size: 14px;
                    box-shadow: 0 4px 14px 0 rgba(234, 88, 12, 0.39);">
                    🖨️ Print / Save as PDF Cookbook
                </button>
                """,
                height=50
            )


# ------------------------------------------
# TAB 2: FRIDGE & PHOTO SCANNER
# ------------------------------------------
with tab2:
    st.markdown("### 📸 Fridge & Food Photo Scanner (Multimodal AI)")
    st.markdown("Upload a photo of your fridge, pantry, or ingredients. ChefAI will analyze the image and generate a custom gourmet recipe!")

    uploaded_file = st.file_uploader(
        "📷 Choose an image of your ingredients or fridge:",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Fridge / Food Image", use_column_width=True)

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                photo_cuisine = st.selectbox("🌍 Preferred Cuisine", ["Any / Fusion", "Italian", "Indian", "Mexican", "Japanese", "Mediterranean", "American"], key="p_cuis")
            with col_p2:
                photo_diet = st.multiselect("🥗 Dietary Preferences", ["High Protein", "Vegetarian", "Vegan", "Gluten-Free", "Keto"], key="p_diet")

            scan_btn = st.button("🔍 Analyze Photo & Generate Recipe", use_container_width=True)

            if scan_btn:
                if not user_api_key:
                    st.error("🔑 Please enter a valid Gemini API Key in the sidebar!")
                else:
                    client, sdk_type, err_msg = get_gemini_client(user_api_key)
                    if not client:
                        st.error(err_msg)
                    else:
                        with st.spinner("🧠 Analyzing photo & identifying ingredients..."):
                            system_prompt = """
                            You are a Master Chef and Multimodal AI Expert.
                            Inspect the provided image, identify all ingredients visible, and create a delicious detailed recipe based strictly on the items in the photo.
                            
                            Return ONLY a valid JSON object:
                            {
                              "recipe_title": "string",
                              "tagline": "string",
                              "prep_time_mins": number,
                              "cook_time_mins": number,
                              "total_time_mins": number,
                              "servings": number,
                              "difficulty": "Easy|Intermediate|Hard",
                              "cuisine": "string",
                              "dietary_tags": ["string"],
                              "nutrition_per_serving": {
                                "calories": number, "protein_g": number, "carbs_g": number, "fat_g": number, "fiber_g": number
                              },
                              "ingredients": [
                                { "item": "string", "amount": "string", "category": "produce|protein|dairy|pantry|spices|other" }
                              ],
                              "instructions": [
                                { "step_number": number, "title": "string", "description": "string" }
                              ],
                              "chef_secret_tips": ["string"],
                              "beverage_pairing": "string"
                            }
                            """
                            prompt = f"Identify all visible food items/ingredients in this photo and create a recipe. Cuisine: {photo_cuisine}. Diet: {', '.join(photo_diet) if photo_diet else 'None'}"

                            try:
                                raw_res = generate_gemini_response(
                                    client=client,
                                    sdk_type=sdk_type,
                                    model_name=selected_model,
                                    prompt=prompt,
                                    image=image,
                                    system_instruction=system_prompt,
                                    temperature=creativity
                                )
                                recipe_data = normalize_recipe_data(parse_json_response(raw_res))
                                st.session_state["current_recipe"] = recipe_data
                                st.success("🎉 Recipe generated from your photo! View full recipe details below or in Tab 1 / Tab 3.")
                            except Exception as e:
                                st.error(f"❌ Failed to process photo: {e}")
        except Exception as img_err:
            st.error(f"⚠️ Error loading image file: {img_err}")


# ------------------------------------------
# TAB 3: HANDS-FREE COOKING STUDIO & AI Q&A ASSISTANT
# ------------------------------------------
with tab3:
    st.markdown("### 👨‍🍳 Hands-Free Cooking Studio & Live AI Assistant")
    st.markdown("Step-by-step cooking view with large text, countdown timers, voice assistant, and live culinary Q&A.")

    # Default fallback sample recipe loader if user hasn't generated one yet
    if "current_recipe" not in st.session_state:
        st.info("💡 Tip: Load a sample recipe below or generate a custom recipe in Tab 1 / Tab 2 to start cooking!")
        sample_choice = st.selectbox(
            "🍳 Select Sample Recipe to Load:",
            ["Creamy Garlic Butter Chicken & Spinach", "Classic Italian Spaghetti Carbonara"],
            key="sample_recipe_picker"
        )
        if st.button("🚀 Load Selected Recipe into Cooking Studio"):
            if "Carbonara" in sample_choice:
                st.session_state["current_recipe"] = {
                    "recipe_title": "Classic Italian Spaghetti Carbonara",
                    "tagline": "Rich, creamy pancetta and egg pasta",
                    "prep_time_mins": 10, "cook_time_mins": 15, "total_time_mins": 25,
                    "servings": 2, "difficulty": "Easy", "cuisine": "Italian", "dietary_tags": ["High Protein"],
                    "nutrition_per_serving": {"calories": 650, "protein_g": 28, "carbs_g": 72, "fat_g": 26, "fiber_g": 3},
                    "ingredients": [
                        {"item": "Spaghetti", "amount": "200g"},
                        {"item": "Pancetta or Bacon", "amount": "100g"},
                        {"item": "Egg Yolks", "amount": "3 large"},
                        {"item": "Pecorino Romano Cheese", "amount": "50g grated"},
                        {"item": "Black Pepper", "amount": "1 tsp freshly cracked"}
                    ],
                    "instructions": [
                        {"step_number": 1, "title": "Boil Pasta", "description": "Bring a large pot of salted water to boil. Cook spaghetti until al dente (about 9-10 mins). Reserve 1/2 cup pasta water."},
                        {"step_number": 2, "title": "Crisp Pancetta", "description": "In a skillet over medium heat, crisp pancetta for 5-6 mins until golden and fat melts. Turn off heat."},
                        {"step_number": 3, "title": "Mix Sauce", "description": "Whisk egg yolks, grated Pecorino Romano, and cracked black pepper together in a small bowl until smooth."},
                        {"step_number": 4, "title": "Combine & Emulsify", "description": "Toss hot drained pasta into skillet with pancetta. Remove skillet from heat. Pour egg mixture over pasta, tossing rapidly with splash of reserved pasta water to create a creamy sauce."}
                    ],
                    "chef_secret_tips": ["Never add eggs while skillet is directly on flame or they will scramble."],
                    "beverage_pairing": "Crisp Pinot Grigio or Italian Sparkling Water"
                }
                if hasattr(st, "rerun"): st.rerun()
            else:
                st.session_state["current_recipe"] = {
                    "recipe_title": "Creamy Garlic Butter Chicken & Spinach",
                    "tagline": "Pan-seared tender chicken breast in garlic parmesan cream",
                    "prep_time_mins": 10, "cook_time_mins": 15, "total_time_mins": 25,
                    "servings": 2, "difficulty": "Easy", "cuisine": "American", "dietary_tags": ["Keto", "High Protein"],
                    "nutrition_per_serving": {"calories": 520, "protein_g": 42, "carbs_g": 8, "fat_g": 34, "fiber_g": 2},
                    "ingredients": [
                        {"item": "Chicken Breast", "amount": "2 fillets"},
                        {"item": "Garlic", "amount": "4 cloves minced"},
                        {"item": "Heavy Cream", "amount": "1/2 cup"},
                        {"item": "Fresh Spinach", "amount": "2 cups"},
                        {"item": "Butter", "amount": "2 tbsp"}
                    ],
                    "instructions": [
                        {"step_number": 1, "title": "Sear Chicken", "description": "Season chicken breasts with salt and pepper. Melt 1 tbsp butter in skillet over medium-high heat. Sear chicken 5-6 mins per side until golden (internal temp 165°F). Transfer to plate."},
                        {"step_number": 2, "title": "Sauté Garlic & Spinach", "description": "Add remaining butter and minced garlic to skillet. Sauté for 1 min until fragrant. Stir in fresh spinach until wilted."},
                        {"step_number": 3, "title": "Simmer Cream Sauce", "description": "Pour in heavy cream and parmesan cheese. Simmer for 2-3 mins until sauce thickens slightly. Return chicken to skillet and spoon sauce over top."}
                    ],
                    "chef_secret_tips": ["Let chicken rest for 3 minutes after sear before slicing for juicy meat."],
                    "beverage_pairing": "Chardonnay or Iced Lemon Tea"
                }
                if hasattr(st, "rerun"): st.rerun()

    if "current_recipe" in st.session_state:
        recipe = normalize_recipe_data(st.session_state["current_recipe"])
        steps = recipe.get("instructions", [])

        if not steps:
            st.warning("No instructions available for current recipe.")
        else:
            st.markdown(f"### 🍽️ {recipe.get('recipe_title')}")
            step_nums = [s.get("step_number", idx + 1) for idx, s in enumerate(steps)]
            selected_step_num = st.selectbox("Select Cooking Step:", step_nums, format_func=lambda x: f"Step {x}: {steps[x-1].get('title', '')}")

            current_step = steps[selected_step_num - 1]

            # Large Focus Card
            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.9); border: 2px solid #ea580c; border-radius: 16px; padding: 28px; margin-top: 15px; text-align: center;">
                <h2 style="color: #fb923c; font-size: 2.2rem; margin-bottom: 10px;">Step {selected_step_num}: {current_step.get('title', '')}</h2>
                <p style="font-size: 1.35rem; color: #f8fafc; line-height: 1.6;">{current_step.get('description', '')}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Audio Text-To-Speech Narration
            step_text = f"Step {selected_step_num}. {current_step.get('title', '')}. {current_step.get('description', '')}"
            st.components.v1.html(
                f"""
                <script>
                    function speakStep() {{
                        if ('speechSynthesis' in window) {{
                            window.speechSynthesis.cancel();
                            var msg = new SpeechSynthesisUtterance({json.dumps(step_text)});
                            msg.rate = 0.95;
                            msg.pitch = 1.0;
                            window.speechSynthesis.speak(msg);
                        }} else {{
                            alert("Text-to-speech is not supported in this browser.");
                        }}
                    }}
                </script>
                <button onclick="speakStep()" style="
                    background: linear-gradient(90deg, #8b5cf6, #ec4899);
                    color: white; font-weight: bold; border: none; padding: 12px 28px;
                    border-radius: 12px; cursor: pointer; font-size: 16px; width: 100%;
                    box-shadow: 0 4px 14px 0 rgba(139, 92, 246, 0.4);">
                    🔊 Read Step Out Loud (Voice Assistant)
                </button>
                """,
                height=60
            )

            st.markdown("---")
            st.markdown("#### ⏱️ Kitchen Countdown Timer")

            timer_c1, timer_c2 = st.columns([1, 2])

            with timer_c1:
                timer_minutes = st.number_input("Set Timer (Minutes):", min_value=1, max_value=120, value=5, key="studio_timer_mins")
                start_timer_btn = st.button("▶️ Start Kitchen Timer", use_container_width=True)

            with timer_c2:
                if start_timer_btn:
                    placeholder = st.empty()
                    for secs in range(int(timer_minutes * 60), -1, -1):
                        mins_left = secs // 60
                        secs_left = secs % 60
                        placeholder.markdown(f"<h1 style='color: #4ade80; font-size: 3.8rem; text-align: center; margin:0;'>⏱️ {mins_left:02d}:{secs_left:02d}</h1>", unsafe_allow_html=True)
                        time.sleep(1)
                    st.balloons()
                    st.success("🔔 Kitchen Timer Finished!")

    # Live Interactive AI Cooking Q&A Section (Available Always)
    st.markdown("---")
    st.markdown("### 💬 Ask ChefAI a Cooking Question (Live AI Assistant)")
    st.markdown("Ask any culinary or cooking question while in the kitchen and get instant expert advice.")

    qa_col1, qa_col2 = st.columns([3, 1])
    with qa_col1:
        user_question = st.text_input(
            "Type your cooking question:",
            placeholder="e.g. Can I substitute heavy cream with milk? How do I tell if chicken is done?",
            key="cooking_qa_input"
        )
    with qa_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        ask_btn = st.button("❓ Ask ChefAI", use_container_width=True)

    if ask_btn and user_question.strip():
        if not user_api_key:
            st.error("🔑 Please enter a valid Gemini API Key in the sidebar!")
        else:
            client, sdk_type, err_msg = get_gemini_client(user_api_key)
            if not client:
                st.error(err_msg)
            else:
                with st.spinner("🧑‍🍳 ChefAI is answering your question..."):
                    current_title = st.session_state.get("current_recipe", {}).get("recipe_title", "General Cooking")
                    sys_prompt = "You are an expert executive chef. Provide concise, clear, practical culinary advice to the user's cooking question."
                    qa_prompt = f"Active Recipe Context: {current_title}\n\nUser Question: {user_question}"

                    try:
                        qa_response = generate_gemini_response(
                            client, sdk_type, selected_model, qa_prompt, image=None, system_instruction=sys_prompt, temperature=creativity
                        )
                        st.markdown("---")
                        st.markdown(f"#### 💡 ChefAI's Answer to: *'{user_question}'*")
                        st.info(qa_response)

                        # Audio narration for Q&A answer
                        st.components.v1.html(
                            f"""
                            <script>
                                function speakAnswer() {{
                                    if ('speechSynthesis' in window) {{
                                        window.speechSynthesis.cancel();
                                        var msg = new SpeechSynthesisUtterance({json.dumps(qa_response)});
                                        msg.rate = 0.95;
                                        window.speechSynthesis.speak(msg);
                                    }}
                                }}
                            </script>
                            <button onclick="speakAnswer()" style="
                                background: linear-gradient(90deg, #10b981, #059669);
                                color: white; font-weight: bold; border: none; padding: 10px 20px;
                                border-radius: 10px; cursor: pointer; font-size: 14px;">
                                🔊 Listen to Answer
                            </button>
                            """,
                            height=50
                        )
                    except Exception as e:
                        st.error(f"❌ Failed to get answer: {e}")


# ------------------------------------------
# TAB 4: WEEKLY MEAL PLANNER
# ------------------------------------------
with tab4:
    st.markdown("### 🗓️ AI Weekly Meal Plan Generator")
    st.markdown("Design a customized multi-day meal plan tailored to your nutritional targets and diet.")

    m_col1, m_col2 = st.columns([1, 1])
    with m_col1:
        plan_days = st.radio("Plan Duration", [3, 7], horizontal=True)
        target_calories = st.number_input("Target Daily Calories", value=2000, step=100)
        meal_diet = st.selectbox("Diet Preference", ["Standard balanced", "High Protein / Fitness", "Keto / Low-Carb", "Vegan", "Vegetarian", "Mediterranean"])
    with m_col2:
        budget_pref = st.selectbox("Budget & Prep Preference", ["Budget Friendly & Quick Prep", "Gourmet Variety", "Minimal Ingredient Re-use"])
        allergies = st.text_input("Allergies or Dislikes", placeholder="e.g., No shellfish, no peanuts")

    plan_btn = st.button("🗓️ Generate Meal Plan", use_container_width=True)

    if plan_btn:
        if not user_api_key:
            st.error("🔑 Please enter a valid Gemini API Key in the sidebar!")
        else:
            client, sdk_type, err_msg = get_gemini_client(user_api_key)
            if not client:
                st.error(err_msg)
            else:
                with st.spinner("🗓️ Generating your custom meal plan..."):
                    system_prompt = f"""
                    You are an expert Executive Nutritionist & Culinary Coach.
                    Generate a COMPLETE {plan_days}-day meal plan.

                    CRITICAL INSTRUCTIONS:
                    1. You MUST generate an entry in the 'days' array for EVERY SINGLE DAY from Day 1 to Day {plan_days}. For a {plan_days}-day plan, there MUST be exactly {plan_days} items in the "days" list.
                    2. For EVERY meal (breakfast, lunch, dinner, snack), provide:
                       - "title": Specific delicious dish name (e.g., "Avocado & Poached Egg Toast"). DO NOT leave title empty.
                       - "calories": Estimated calorie count as a number (e.g., 450).
                       - "brief": A 1-2 sentence description of ingredients and quick prep method.

                    Return ONLY raw JSON matching this schema:
                    {{
                      "plan_title": "string",
                      "daily_summary": "string",
                      "days": [
                        {{
                          "day_number": 1,
                          "day_name": "Day 1",
                          "breakfast": {{ "title": "string", "calories": 400, "brief": "string" }},
                          "lunch": {{ "title": "string", "calories": 600, "brief": "string" }},
                          "dinner": {{ "title": "string", "calories": 700, "brief": "string" }},
                          "snack": {{ "title": "string", "calories": 300, "brief": "string" }}
                        }}
                      ]
                    }}
                    """
                    user_prompt = f"""
                    Please generate a full {plan_days}-day meal plan!
                    - Total Days Required: {plan_days}
                    - Target Daily Calories: {target_calories} kcal
                    - Diet Preference: {meal_diet}
                    - Budget & Prep Style: {budget_pref}
                    - Allergies / Dislikes: {allergies if allergies else 'None'}

                    CRITICAL: You MUST include ALL {plan_days} days in the 'days' array with non-empty titles and descriptions!
                    """

                    try:
                        raw = generate_gemini_response(
                            client, sdk_type, selected_model, user_prompt, system_prompt, creativity
                        )
                        plan_data = normalize_meal_plan(parse_json_response(raw))
                        st.session_state["meal_plan"] = plan_data
                        st.success("✅ Meal plan generated!")
                    except Exception as e:
                        st.error(f"❌ Failed to generate meal plan: {e}")

    if "meal_plan" in st.session_state:
        plan = normalize_meal_plan(st.session_state["meal_plan"])
        st.session_state["meal_plan"] = plan

        st.markdown("---")
        st.markdown(f"### 📋 {plan.get('plan_title', 'Custom Meal Plan')}")
        st.caption(plan.get("daily_summary", ""))

        days_list = plan.get("days", [])
        if isinstance(days_list, list):
            for idx, d in enumerate(days_list):
                if not isinstance(d, dict):
                    continue
                day_num = d.get("day_number", idx + 1)
                day_name = d.get("day_name", f"Day {day_num}")

                with st.expander(f"📌 Day {day_num}: {day_name}", expanded=True):
                    dc1, dc2, dc3, dc4 = st.columns(4)
                    for col, meal_key, meal_label, icon in [
                        (dc1, "breakfast", "Breakfast", "🍳"),
                        (dc2, "lunch", "Lunch", "🥗"),
                        (dc3, "dinner", "Dinner", "🍽️"),
                        (dc4, "snack", "Snack", "🍎")
                    ]:
                        with col:
                            m = d.get(meal_key, {})
                            if isinstance(m, dict):
                                title = m.get("title", m.get("name", "Meal Item"))
                                cal = m.get("calories", m.get("kcal", 0))
                                brief = m.get("brief", m.get("description", ""))
                                st.markdown(f"**{icon} {meal_label}** ({cal} kcal)")
                                st.write(f"*{title}*: {brief}")
                            elif isinstance(m, str):
                                st.markdown(f"**{icon} {meal_label}**")
                                st.write(m)


# ------------------------------------------
# TAB 5: RECIPE TRANSFORMER
# ------------------------------------------
with tab5:
    st.markdown("### 🔄 Transform Any Recipe")
    st.markdown("Paste an existing recipe and ask ChefAI to modify it (e.g. make it Vegan, air-fryer compatible, or low-sodium).")

    original_recipe = st.text_area("Paste Original Recipe Text:", height=150, placeholder="Paste ingredients and instructions here...")
    transformation_goal = st.text_input("What transformation would you like?", placeholder="e.g. Make this gluten-free and air-fryer friendly")

    transform_btn = st.button("✨ Transform Recipe")

    if transform_btn:
        if not user_api_key:
            st.error("🔑 Please enter a valid Gemini API Key!")
        elif not original_recipe.strip() or not transformation_goal.strip():
            st.warning("⚠️ Please fill in both the original recipe and the transformation goal.")
        else:
            client, sdk_type, err_msg = get_gemini_client(user_api_key)
            if not client:
                st.error(err_msg)
            else:
                with st.spinner("🔄 Re-engineering recipe..."):
                    system_prompt = "You are a master culinary scientist. Transform the input recipe as requested and return clear formatted markdown."
                    prompt = f"Original Recipe:\n{original_recipe}\n\nTransformation Request: {transformation_goal}"

                    try:
                        resp = generate_gemini_response(
                            client, sdk_type, selected_model, prompt, image=None, system_instruction=system_prompt, temperature=creativity
                        )
                        st.markdown("---")
                        st.markdown("### 🌟 Transformed Recipe")
                        st.markdown(resp)
                    except Exception as e:
                        st.error(f"❌ Transformation failed: {e}")


# ------------------------------------------
# TAB 6: SHOPPING LIST HUB
# ------------------------------------------
with tab6:
    st.markdown("### 🛒 Smart Grocery & Pantry List")
    st.markdown("Items from generated recipes automatically populate here or add items manually.")

    if "current_recipe" in st.session_state:
        recipe_ing = st.session_state["current_recipe"].get("ingredients", [])
        st.markdown(f"**Recipe:** {st.session_state['current_recipe'].get('recipe_title')}")
        
        for ing in recipe_ing:
            st.checkbox(f"{ing.get('amount')} {ing.get('item')} ({ing.get('category', 'general').capitalize()})", value=False)
    else:
        st.info("💡 Generate a recipe in the 'Instant Recipe Generator' tab to auto-populate your shopping list!")

    st.markdown("---")
    st.markdown("#### ➕ Quick Add Manual Items")
    new_item = st.text_input("Add Item:")
    if st.button("Add to List") and new_item:
        st.success(f"Added {new_item}!")

