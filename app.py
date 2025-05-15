import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# تحميل النموذج والمحول
model = joblib.load("alz_model.pkl")
scaler = joblib.load("scaler.pkl")

# إعداد الصفحة
st.set_page_config(page_title="Alzheimer's Prediction", layout="centered")

# إضافة CSS للأنيميشن والخلفية وللزر في الوسط مع حركة
st.markdown("""
<style>
    /* خلفية داكنة */
    .stApp {
        background: linear-gradient(135deg, #1f2937, #111827);
        color: #f0f4f8;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    /* صندوق المحتوى */
    .block-container {
        background-color: rgba(30, 41, 59, 0.85);
        padding: 2.5rem 3rem;
        border-radius: 18px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.6);
        color: #f0f4f8;
        max-width: 700px;
        margin: auto;
    }
    /* العنوان الكبير بالوسط مع انميشن نبض */
    .animated-title {
        text-align: center;
        font-size: 3.5rem;
        font-weight: 900;
        color: #e1eaeb;
        text-shadow:
            0 0 5px #7ba3d4,
            0 0 10px #22d3ee,
            0 0 20px #06b6d4,
            0 0 40px #0e7490;
        animation: pulseGlow 3s ease-in-out infinite;
        margin-bottom: 0.2rem;
    }
    @keyframes pulseGlow {
        0%, 100% {
            transform: scale(1);
            text-shadow:
                0 0 5px #67e8f9,
                0 0 10px #22d3ee,
                0 0 20px #06b6d4,
                0 0 40px #0e7490;
        }
        50% {
            transform: scale(1.1);
            text-shadow:
                0 0 15px #67e8f9,
                0 0 30px #22d3ee,
                0 0 40px #06b6d4,
                0 0 70px #0e7490;
        }
    }
    /* العناوين الفرعية */
    h3 {
        text-align: center;
        color: #bae6fd;
        margin-bottom: 2rem;
        font-weight: 600;
    }
    /* زرار ستريمليت */
    .stButton>button {
        background: linear-gradient(90deg, #2563eb, #3b82f6);
        border: none;
        padding: 0.7rem 2.2rem;
        border-radius: 10px;
        font-weight: 700;
        color: white;
        font-size: 1.1rem;
        transition: background 0.3s ease;
        display: block;
        margin: 1.5rem auto; /* بالضبط في الوسط */
        animation: buttonPulse 2.5s ease-in-out infinite;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #1e40af, #2563eb);
        cursor: pointer;
    }
    @keyframes buttonPulse {
        0%, 100% {
            transform: scale(1);
            box-shadow: 0 0 10px #3b82f6;
        }
        50% {
            transform: scale(1.05);
            box-shadow: 0 0 25px #60a5fa;
        }
    }
    /* نصوص الإدخال بالعربي */
    label, .css-1n76uvr {
        color: #cbd5e1 !important;
        font-weight: 600;
    }
    /* مظهر السلايدر */
    .stSlider>div>div>div>input {
        color: #f0f4f8 !important;
    }
</style>
""", unsafe_allow_html=True)

# العنوان بالإنجليزي فقط وبالوسط مع الحركة
st.markdown('<h1 class="animated-title">Alzheimer\'s Disease Prediction</h1>', unsafe_allow_html=True)
st.markdown("<h3>Please fill in the information below to assess your risk of Alzheimer's.</h3>", unsafe_allow_html=True)
st.write("---")

# نموذج الإدخال بالعربي
with st.form(key="alz_form"):
    st.write("## البيانات الأساسية")
    age = st.number_input("العمر:", min_value=40, max_value=100, step=1)
    gender = st.selectbox("النوع:", ["ذكر", "أنثى"])

    st.write("## العوامل الصحية")
    smoking = st.selectbox("هل تدخن؟", ["لا", "نعم"])
    alcohol = st.slider("معدل استهلاك الكحول (0-10):", 0, 10, 5)
    physical_activity = st.slider("النشاط البدني (0-10):", 0, 10, 5)
    diet_quality = st.slider("جودة التغذية (0-10):", 0, 10, 5)
    sleep_quality = st.slider("جودة النوم (0-10):", 0, 10, 5)
    family_history = st.selectbox("هل يوجد تاريخ عائلي للزهايمر؟", ["لا", "نعم"])
    cardiovascular = st.selectbox("هل تعاني من أمراض القلب؟", ["لا", "نعم"])
    diabetes = st.selectbox("هل تعاني من السكري؟", ["لا", "نعم"])
    depression = st.selectbox("هل تعاني من الاكتئاب؟", ["لا", "نعم"])
    head_injury = st.selectbox("هل سبق وتعرضت لإصابة في الرأس؟", ["لا", "نعم"])
    hypertension = st.selectbox("هل تعاني من ضغط الدم المرتفع؟", ["لا", "نعم"])

    st.write("## التقييم الإدراكي والسلوكي")
    mmse = st.slider("اختبار الحالة الذهنية MMSE (0 = ضعف شديد إلى 30 = طبيعي):", 0, 30, 15)
    functional_assessment = st.slider("تقييم الوظائف الإدراكية (0-10):", 0, 10, 5)
    memory_complaints = st.selectbox("هل تعاني من مشاكل في الذاكرة؟", ["لا", "نعم"])
    behavioral_problems = st.selectbox("هل تعاني من مشاكل سلوكية؟", ["لا", "نعم"])
    adl = st.slider("القدرة على أداء المهام اليومية (0-10):", 0, 10, 5)
    confusion = st.selectbox("هل تعاني من الارتباك؟", ["لا", "نعم"])
    disorientation = st.selectbox("هل تعاني من فقدان الاتجاه؟", ["لا", "نعم"])
    personality_changes = st.selectbox("هل هناك تغيّرات في الشخصية؟", ["لا", "نعم"])
    difficulty_tasks = st.selectbox("هل تواجه صعوبة في إنجاز المهام؟", ["لا", "نعم"])
    forgetfulness = st.selectbox("هل تنسى كثيرًا؟", ["لا", "نعم"])

    submit = st.form_submit_button("Cognitive Status Report")

# معالجة الإدخال وعرض النتائج
if submit:
    input_data = np.array([
        age,
        1 if gender == "ذكر" else 0,
        1 if smoking == "نعم" else 0,
        alcohol,
        physical_activity,
        diet_quality,
        sleep_quality,
        1 if family_history == "نعم" else 0,
        1 if cardiovascular == "نعم" else 0,
        1 if diabetes == "نعم" else 0,
        1 if depression == "نعم" else 0,
        1 if head_injury == "نعم" else 0,
        1 if hypertension == "نعم" else 0,
        mmse,
        functional_assessment,
        1 if memory_complaints == "نعم" else 0,
        1 if behavioral_problems == "نعم" else 0,
        adl,
        1 if confusion == "نعم" else 0,
        1 if disorientation == "نعم" else 0,
        1 if personality_changes == "نعم" else 0,
        1 if difficulty_tasks == "نعم" else 0,
        1 if forgetfulness == "نعم" else 0,
    ]).reshape(1, -1)

    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)[0]

    st.write("---")
    if prediction == 1:
        st.markdown("<h3 style='text-align: center; color: #ff6b6b;'> قد تكون مصابًا بمرض الزهايمر. يُرجى استشارة الطبيب.</h3>", unsafe_allow_html=True)
        st.info("""
        - راجع طبيب مختص في أسرع وقت.
        - مارس تمارين تنشيط الدماغ.
        - اتبع نظام غذائي صحي.
        - اهتم بالنوم والنشاط البدني.
        - احصل على دعم عائلي ونفسي.
        """)
    else:
        st.markdown("<h3 style='text-align: center; color: #4ade80;'>✅ لا يوجد مؤشر واضح للإصابة بالزهايمر.</h3>", unsafe_allow_html=True)
        st.success("""
        - استمر في أسلوب حياة صحي.
        - قم بتمارين ذهنية يومية.
        - تابع نفسك بشكل دوري.
        """)

    # شرح عربي قبل الرسم البياني
    st.subheader("شرح نتائج التقييم الإدراكي")
    st.markdown("""
    هذا الرسم البياني يوضح نتائجك في المؤشرات الإدراكية الرئيسية مقارنة بالنتائج المثالية .
    """)

    # رسم بياني بالإنجليزي
    features = ['MMSE', 'Cognitive Function', 'ADL']
    values = [mmse, functional_assessment, adl]
    ideal = [30, 10, 10]

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_width = 0.4
    x = np.arange(len(features))

    ax.bar(x - bar_width/2, values, bar_width, label='Your Scores', color='#22d3ee')
    ax.bar(x + bar_width/2, ideal, bar_width, label='Ideal Scores', color='#a5b4fc')

    ax.set_xlabel('Indicators', fontsize=12)
    ax.set_ylabel('Scores', fontsize=12)
    ax.set_title('Your Scores vs Ideal Scores', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(features, fontsize=11)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    st.pyplot(fig)