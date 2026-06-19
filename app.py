import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# 1. MUST BE FIRST: Page Configuration
st.set_page_config(page_title="Waste Classifier Pro", layout="wide")

# 2. Language Dictionary Definition
languages = {
    "日本語": {
        "title": "♻️ ゴミ分別AIアプリ",
        "sidebar": "👤 ユーザー情報",
        "name": "お名前",
        "age": "年齢",
        "address": "ご住所",
        "btn": "情報を保存する",
        "radio": "ゴミの画像入力方法を選択してください:",
        "options": ["カメラで撮影", "画像をアップロード"],
        "success": "情報を保存しました！",
        "cam_caption": "カメラで撮影してください",
        "file_caption": "画像をアップロードしてください",
        "result_title": "📊 ゴミの分類結果",
        "welcome": "さん、ご利用ありがとうございます。正しく分別できるようにサポートいたします。",
        "cal_title": "📅 ゴミ収集日カレンダー",
        "cal_select": "ゴミの種類を選択:",
        "cal_days": "収集日"
    },
    "Burmese": {
        "title": "♻️ အမှိုက်(ရည်းစား‌ဟောင်း)ခွဲခြားပေးသည့် AI App",
        "sidebar": "👤 အသုံးပြုသူ အချက်အလက်",
        "name": "အမည်",
        "age": "အသက်",
        "address": "နေရပ်လိပ်စာ",
        "btn": "အချက်အလက် သိမ်းမည်",
        "radio": "အမှိုက်ပုံကို မည်သို့ ထည့်သွင်းမည်နည်း:",
        "options": ["ကင်မရာဖြင့် ရိုက်မည်", "ဓာတ်ပုံ တင်မည်"],
        "success": "အချက်အလက် သိမ်းပြီးပါပြီ!",
        "cam_caption": "ကင်မရာဖြင့် ဓာတ်ပုံရိုက်ပါ",
        "file_caption": "ဓာတ်ပုံတင်ပေးပါ",
        "result_title": "📊 အမှိုက်အမျိုးအစား ခွဲခြားမှုရလဒ်",
        "welcome": "ရေ၊ အသုံးပြုမှုအတွက် ကျေးဇူးတင်ပါသည်။ မှန်ကန်စွာ ခွဲခြားနိုင်ရန် ကူညီပေးပါမည်။",
        "cal_title": "📅 အမှိုက်သိမ်းရက် ပြက္ခဒိန်",
        "cal_select": "အမှိုက်အမျိုးအစား ရွေးချယ်ရန်:",
        "cal_days": "အမှိုက်သိမ်းရက်"
    }
}

# Language Selector (Sidebar Top)
lang_key = st.sidebar.selectbox("Language / ဘာသာစကား", ["日本語", "Burmese"])
lang = languages[lang_key]

# Display Main Title
st.title(lang["title"])
# 3. Model Loading (Cached)
@st.cache_resource
def load_model(lang_selection):
    interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
    interpreter.allocate_tensors()
    # Choose label file based on language
    label_file = 'labels.txt' if lang_selection == "日本語" else 'labels_my.txt'
    try:
        with open(label_file, 'r', encoding='utf-8') as f:
            labels = f.readlines()
    except FileNotFoundError:
        # Fallback if specific file missing
        with open('labels.txt', 'r', encoding='utf-8') as f:
            labels = f.readlines()
    return interpreter, labels

interpreter, labels = load_model(lang_key)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 4. User Info Sidebar
with st.sidebar:
    st.header(lang["sidebar"])
    name = st.text_input(lang["name"])
    age = st.number_input(lang["age"], min_value=1, max_value=100)
    address = st.text_area(lang["address"])
    if st.button(lang["btn"]):
        st.success(lang["success"])

# 5. Feature: Garbage Calendar
def show_garbage_calendar():
    st.sidebar.markdown("---")
    st.sidebar.subheader(lang["cal_title"])
    
    # Schedule data mapping
    if lang_key == "日本語":
        schedule = {
            "燃えるゴミ (Burnable)": ["月曜日", "木曜日"],
            "資源ゴミ (Resources)": ["水曜日"],
            "粗大ゴミ (Large Items)": ["要予約 (要連絡)"]
        }
    else:
        schedule = {
            "မီးရှို့နိုင်သော အမှိုက် (Burnable)": ["တနင်္လာနေ့", "ကြာသပတေးနေ့"],
            "ပြန်လည်အသုံးပြုနိုင်သော အမှိုက် (Resources)": ["ဗုဒ္ဓဟူးနေ့"],
            "အမှိုက်ကြီးများ (Large Items)": ["ကြိုတင်ဘိုကင်လုပ်ရန် လိုအပ်သည်"]
        }
    
    selected_type = st.sidebar.selectbox(lang["cal_select"], list(schedule.keys()))
    st.sidebar.write(f"{lang['cal_days']}: {', '.join(schedule[selected_type])}")

show_garbage_calendar()

# 6. Image Input Method Selection
choice = st.radio(lang["radio"], lang["options"])

image = None
if choice == lang["options"][0]: 
    image = st.camera_input(lang["cam_caption"])
else: 
    image = st.file_uploader(lang["file_caption"], type=['jpg', 'jpeg', 'png'])

# 7. AI Processing & Results
if image:
    # Read image using PIL and convert to format OpenCV expects
    img_data = Image.open(image).convert('RGB')
    img_array = np.array(img_data)
    
    # Process image for Teachable Machine Model (224x224)
    processed_img = cv2.resize(img_array, (224, 224))
    processed_img = np.expand_dims(processed_img, axis=0).astype(np.float32)
    
    # Teachable Machine normalization: (img / 127.5) - 1
    processed_img = (processed_img / 127.5) - 1
    
    # Run Inference
    interpreter.set_tensor(input_details[0]['index'], processed_img)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    highest_match_index = np.argmax(prediction)
    
    # Display Results Side-by-Side or Block layout
    st.image(image, caption=lang["options"][1], width=400)
    st.subheader(lang["result_title"])
    
    for i, label in enumerate(labels):
        conf = prediction[0][i] * 100
        status = "✅" if i == highest_match_index else "❌"
        st.write(f"{status} {label.strip()} ({conf:.2f}%)")
        st.progress(float(prediction[0][i]))

    # Personalized Greeting if Name is given
    if name:
        st.info(f"{name}{lang['welcome']}")
        # ၁။ Import များနဲ့ အခြား Setup များ
import streamlit as st
import random
import base64

def play_sound(file_name):
    # အသံဖိုင်ရှိမရှိ စစ်ဆေးပြီး အသံဖွင့်ပေးခြင်း
    try:
        with open(f"sounds/{file_name}", "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            audio_html = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{b64}" type="audio/mpeg"></audio>'
            st.markdown(audio_html, unsafe_allow_html=True)
    except:
        pass # အသံဖိုင်မရှိရင် error မတက်အောင်ထားခြင်း

if 'game_status' not in st.session_state:
    st.session_state.game_status = "idle"
    st.session_state.score = 0
    st.session_state.question_count = 0

waste_data = {
    "🍌 ငှက်ပျောသီးအခွံ": "🔥",
    "📰 သတင်းစာ": "🔥",
    "🪵 သစ်ကိုင်းခြောက်": "🔥",
    "👕 အဝတ်စုတ်": "🔥",
    "🔋 ဘက်ထရီ": "🧊",
    "🥫 သံဘူး": "🧊",
    "🍷 မှန်ကွဲ": "🧊",
    "💡 မီးသီးအဟောင်း": "🧊",
    "🏺 ကြွေထည်": "🧊"
}

def start_game():
    st.session_state.game_status = "playing"
    st.session_state.score = 0
    st.session_state.question_count = 0
    st.session_state.trash = random.choice(list(waste_data.keys()))

st.title("♻️ အမှိုက်ကောက်ဂိမ်း")

if st.session_state.game_status == "idle":
    if st.button("ဂိမ်းစရန် နှိပ်ပါ"):
        start_game()
        st.rerun()

elif st.session_state.game_status == "playing":
    if st.session_state.question_count >= 10:
        st.subheader("🎉 ဂိမ်းပြီးဆုံးပါပြီ!")
        st.write(f"### သင်၏ စုစုပေါင်းရမှတ်: {st.session_state.score} မှတ်")
        
        # ဆက်ဆော့မလား/ရပ်မလား ရွေးချယ်ခွင့်
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🔄 ထပ်ဆော့မည်"):
                start_game()
                st.rerun()
        with col_b:
            if st.button("❌ ဂိမ်းမှထွက်မည်"):
                st.session_state.game_status = "idle"
                st.rerun()
    else:
        st.write(f"### မေးခွန်း: {st.session_state.question_count + 1} / 10")
        st.write(f"### ဒီအရာကို ဘယ်ထဲထည့်မလဲ?: {st.session_state.trash}")
        
        col1, col2 = st.columns(2)
        
        def check_answer(user_choice):
            if waste_data[st.session_state.trash] == user_choice:
                play_sound("correct.mp3") # မှန်လျှင် အသံ
                st.session_state.score += 10
                st.success("မှန်တယ်! (+10 မှတ်)")
            else:
                play_sound("wrong.mp3") # မှားလျှင် အသံ
                st.error("မှားတယ်!")
            
            st.session_state.question_count += 1
            if st.session_state.question_count < 10:
                st.session_state.trash = random.choice(list(waste_data.keys()))
            st.rerun()

        with col1:
            if st.button("🔥 လောင်ကျွမ်းနိုင်သည်"): check_answer("🔥")
        with col2:
            if st.button("🧊 မလောင်ကျွမ်းနိုင်ပါ"): check_answer("🧊")