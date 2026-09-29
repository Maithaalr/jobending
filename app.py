import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
from io import BytesIO

# ============================================================
# إعداد الصفحة
# ============================================================
st.set_page_config(
    page_title="تحليل المنتهية خدماتهم",
    page_icon="📊",
    layout="wide"
)

st.title("📊 لوحة تحليل المنتهية خدماتهم")
st.caption(
    "تنظيف البيانات تلقائياً، تحليل حالات انتهاء الخدمة، "
    "الاستقالات، مدة الخدمة، الرواتب والاتجاهات الزمنية."
)

# ============================================================
# أسماء الحقول المتوقعة
# ============================================================
EXPECTED_COLUMNS = [
    "رمز الدائرة",
    "الرقم الوظيفي",
    "اسم الموظف",
    "الوحدة التنظيمية",
    "مجموع الراتب",
    "الراتب الاساسي",
    "تاريخ التعيين",
    "تاريخ انتهاء الخدمة",
    "الدرجة الوظيفية",
    "الجنسية",
    "فئة الجنسية",
    "المسمى الوظيفي",
    "سبب انتهاء الخدمة",
    "سبب الاستقالة",
    "الفئة الوظيفية",
    "المجموعة الوظيفية الرئيسية",
    "المجموعة الوظيفية الفرعية",
]

# ============================================================
# دوال مساعدة
# ============================================================
ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
ENGLISH_DIGITS = "0123456789"

TRANSLATION_TABLE = str.maketrans(
    ARABIC_DIGITS + "٫٬",
    ENGLISH_DIGITS + ".,"
)

HIDDEN_CHARS_PATTERN = r"[\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]"


def normalize_column_name(col):
    """تنظيف اسم العمود."""
    col = str(col)
    col = re.sub(HIDDEN_CHARS_PATTERN, "", col)
    col = col.replace("\xa0", " ")
    col = re.sub(r"\s+", " ", col)
    return col.strip()


def normalize_text(value):
    """تنظيف النصوص من المحارف المخفية والمسافات."""
    if pd.isna(value):
        return np.nan

    value = str(value)
    value = re.sub(HIDDEN_CHARS_PATTERN, "", value)
    value = value.replace("\xa0", " ")
    value = re.sub(r"\s+", " ", value)
    value = value.strip()

    if value.lower() in ["", "nan", "none", "null"]:
        return np.nan

    return value


def arabic_to_english_numbers(value):
    """تحويل الأرقام العربية إلى إنجليزية."""
    if pd.isna(value):
        return value

    return str(value).translate(TRANSLATION_TABLE)


def clean_date(value):
    """تنظيف وتحويل التاريخ."""
    if pd.isna(value):
        return pd.NaT

    value = arabic_to_english_numbers(value)
    value = re.sub(HIDDEN_CHARS_PATTERN, "", str(value))
    value = value.replace("\xa0", " ").strip()

    if value.lower() in ["", "0", "nan", "none", "null", "-", "--"]:
        return pd.NaT

    # محاولة مباشرة dd/mm/yyyy
    date_value = pd.to_datetime(
        value,
        format="%d/%m/%Y",
        errors="coerce"
    )

    # إذا لم ينجح، محاولة مرنة
    if pd.isna(date_value):
        date_value = pd.to_datetime(
            value,
            dayfirst=True,
            errors="coerce"
        )

    return date_value


def clean_salary(value):
    """تنظيف الراتب وتحويله إلى رقم."""
    if pd.isna(value):
        return np.nan

    value = arabic_to_english_numbers(value)
    value = re.sub(HIDDEN_CHARS_PATTERN, "", str(value))
    value = value.replace("\xa0", "")
    value = value.replace(" ", "")
    value = value.replace(",", "")

    # إبقاء الأرقام والنقطة والسالب فقط
    value = re.sub(r"[^0-9.\-]", "", value)

    if value in ["", ".", "-", "-."]:
        return np.nan

    try:
        return float(value)
    except:
        return np.nan


def format_number(value, decimals=0):
    if pd.isna(value):
        return "—"

    return f"{value:,.{decimals}f}"


def safe_mode(series):
    series = series.dropna()

    if len(series) == 0:
        return "—"

    modes = series.mode()

    if len(modes) == 0:
        return "—"

    return str(modes.iloc[0])


def count_unique_employees(data):
    """عدد الموظفين الفريدين إذا توفر الرقم الوظيفي."""
    if "الرقم الوظيفي" in data.columns:
        s = data["الرقم الوظيفي"].dropna()

        if len(s) > 0:
            return s.nunique()

    return len(data)


def find_resignation_mask(data):
    """
    تحديد حالات الاستقالة بشكل مرن.
    يبحث عن كلمة استقال في سبب انتهاء الخدمة.
    """
    if "سبب انتهاء الخدمة" not in data.columns:
        return pd.Series(False, index=data.index)

    s = data["سبب انتهاء الخدمة"].fillna("").astype(str)

    return s.str.contains("استقال", case=False, regex=False)


def calculate_service_fields(data):
    """حساب مدة الخدمة."""
    data = data.copy()

    if (
        "تاريخ التعيين" in data.columns
        and "تاريخ انتهاء الخدمة" in data.columns
    ):
        data["مدة الخدمة بالأيام"] = (
            data["تاريخ انتهاء الخدمة"]
            - data["تاريخ التعيين"]
        ).dt.days

        # القيم السالبة تعتبر غير صحيحة
        data.loc[
            data["مدة الخدمة بالأيام"] < 0,
            "مدة الخدمة بالأيام"
        ] = np.nan

        data["مدة الخدمة بالسنوات"] = (
            data["مدة الخدمة بالأيام"] / 365.25
        ).round(2)

        bins = [
            -0.001,
            1,
            3,
            5,
            10,
            15,
            np.inf
        ]

        labels = [
            "أقل من سنة",
            "1 - 3 سنوات",
            "3 - 5 سنوات",
            "5 - 10 سنوات",
            "10 - 15 سنة",
            "أكثر من 15 سنة"
        ]

        data["فئة مدة الخدمة"] = pd.cut(
            data["مدة الخدمة بالسنوات"],
            bins=bins,
            labels=labels,
            right=False
        )

    return data


def add_time_fields(data):
    """إضافة السنة والشهر من تاريخ انتهاء الخدمة."""
    data = data.copy()

    if "تاريخ انتهاء الخدمة" in data.columns:
        data["سنة انتهاء الخدمة"] = (
            data["تاريخ انتهاء الخدمة"].dt.year
        )

        data["رقم شهر انتهاء الخدمة"] = (
            data["تاريخ انتهاء الخدمة"].dt.month
        )

        data["شهر انتهاء الخدمة"] = (
            data["تاريخ انتهاء الخدمة"]
            .dt.to_period("M")
            .astype(str)
        )

    return data


def build_count_table(data, column):
    """جدول العدد والنسبة."""
    if column not in data.columns:
        return pd.DataFrame()

    temp = data[column].fillna("غير محدد")

    table = (
        temp.value_counts(dropna=False)
        .rename_axis(column)
        .reset_index(name="العدد")
    )

    total = table["العدد"].sum()

    if total > 0:
        table["النسبة %"] = (
            table["العدد"] / total * 100
        ).round(2)
    else:
        table["النسبة %"] = 0

    return table


def create_excel_download(sheets):
    """إنشاء ملف Excel من عدة جداول."""
    output = BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        for sheet_name, table in sheets.items():

            if table is None:
                continue

            if not isinstance(table, pd.DataFrame):
                continue

            safe_name = str(sheet_name)[:31]

            export_df = table.copy()

            # Excel لا يدعم timezone
            for col in export_df.columns:
                if pd.api.types.is_datetime64_any_dtype(
                    export_df[col]
                ):
                    export_df[col] = export_df[col].dt.strftime(
                        "%d/%m/%Y"
                    )

            export_df.to_excel(
                writer,
                sheet_name=safe_name,
                index=False
            )

    output.seek(0)
    return output


def show_bar_chart(
    table,
    category,
    value="العدد",
    title="",
    horizontal=False
):
    if table.empty:
        st.info("لا توجد بيانات كافية لعرض الرسم.")
        return

    chart_data = table.copy()

    if horizontal:
        chart_data = chart_data.sort_values(
            value,
            ascending=True
        )

        fig = px.bar(
            chart_data,
            x=value,
            y=category,
            orientation="h",
            text=value,
            title=title
        )
    else:
        fig = px.bar(
            chart_data,
            x=category,
            y=value,
            text=value,
            title=title
        )

    fig.update_layout(
        height=500,
        xaxis_title="",
        yaxis_title=""
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"bar_{title}_{category}_{value}"
    )


# ============================================================
# رفع الملف
# ============================================================
uploaded_file = st.file_uploader(
    "📂 ارفعي تقرير المنتهية خدماتهم",
    type=["xlsx", "xls"]
)

if uploaded_file is None:
    st.info("ارفعي ملف Excel لبدء التحليل.")
    st.stop()


# ============================================================
# اختيار Sheet
# ============================================================
try:
    excel_file = pd.ExcelFile(uploaded_file)

except Exception as e:
    st.error(f"تعذر قراءة ملف Excel: {e}")
    st.stop()


if len(excel_file.sheet_names) > 1:
    selected_sheet = st.selectbox(
        "اختر ورقة البيانات",
        excel_file.sheet_names
    )
else:
    selected_sheet = excel_file.sheet_names[0]


try:
    df_raw = pd.read_excel(
        uploaded_file,
        sheet_name=selected_sheet
    )

except Exception as e:
    st.error(f"حدث خطأ أثناء قراءة البيانات: {e}")
    st.stop()


# ============================================================
# تنظيف أسماء الأعمدة
# ============================================================
df_raw.columns = [
    normalize_column_name(c)
    for c in df_raw.columns
]

# إزالة الصفوف الفارغة بالكامل
df_raw = df_raw.dropna(how="all").copy()

original_row_count = len(df_raw)

# ============================================================
# معالجة اختلاف بسيط في أسماء الأعمدة
# ============================================================
COLUMN_ALIASES = {
    "الراتب الأساسي": "الراتب الاساسي",
    "الراتب الأساسي ": "الراتب الاساسي",
    "رمز الدائرة ": "رمز الدائرة",
    "الفئة الوظيفية ": "الفئة الوظيفية",
    "المجموعة الوظيفية الرئيسية ": "المجموعة الوظيفية الرئيسية",
    "المجموعة الوظيفية الفرعية ": "المجموعة الوظيفية الفرعية",
}

rename_dict = {}

for old, new in COLUMN_ALIASES.items():
    if old in df_raw.columns and new not in df_raw.columns:
        rename_dict[old] = new

df_raw = df_raw.rename(columns=rename_dict)

# ============================================================
# التحقق من الحقول
# ============================================================
missing_columns = [
    c for c in EXPECTED_COLUMNS
    if c not in df_raw.columns
]

if missing_columns:
    with st.expander(
        "⚠️ حقول غير موجودة في الملف",
        expanded=False
    ):
        st.write(
            "سيستمر التحليل بالحقول المتوفرة، "
            "ولكن بعض المؤشرات قد لا تظهر."
        )

        st.write(missing_columns)

# ============================================================
# تنظيف البيانات
# ============================================================
df = df_raw.copy()

# تنظيف النصوص
TEXT_COLUMNS = [
    "رمز الدائرة",
    "الرقم الوظيفي",
    "اسم الموظف",
    "الوحدة التنظيمية",
    "الدرجة الوظيفية",
    "الجنسية",
    "فئة الجنسية",
    "المسمى الوظيفي",
    "سبب انتهاء الخدمة",
    "سبب الاستقالة",
    "الفئة الوظيفية",
    "المجموعة الوظيفية الرئيسية",
    "المجموعة الوظيفية الفرعية",
]

for col in TEXT_COLUMNS:
    if col in df.columns:
        df[col] = df[col].apply(normalize_text)

# حفظ النص الأصلي للتواريخ لفحص الجودة
if "تاريخ التعيين" in df.columns:
    original_hire_dates = df["تاريخ التعيين"].copy()
    df["تاريخ التعيين"] = df["تاريخ التعيين"].apply(
        clean_date
    )
else:
    original_hire_dates = pd.Series(dtype="object")

if "تاريخ انتهاء الخدمة" in df.columns:
    original_end_dates = df["تاريخ انتهاء الخدمة"].copy()
    df["تاريخ انتهاء الخدمة"] = (
        df["تاريخ انتهاء الخدمة"].apply(clean_date)
    )
else:
    original_end_dates = pd.Series(dtype="object")

# تنظيف الرواتب
for salary_col in [
    "مجموع الراتب",
    "الراتب الاساسي"
]:
    if salary_col in df.columns:
        df[salary_col] = df[salary_col].apply(
            clean_salary
        )

# حساب مدة الخدمة
df = calculate_service_fields(df)

# إضافة الحقول الزمنية
df = add_time_fields(df)

# ============================================================
# فحص جودة البيانات
# ============================================================
invalid_hire_dates = 0
invalid_end_dates = 0

if "تاريخ التعيين" in df.columns:
    valid_original = (
        original_hire_dates.notna()
        & ~original_hire_dates.astype(str)
        .str.strip()
        .isin(["", "0", "٠", "nan"])
    )

    invalid_hire_dates = (
        valid_original
        & df["تاريخ التعيين"].isna()
    ).sum()

if "تاريخ انتهاء الخدمة" in df.columns:
    valid_original = (
        original_end_dates.notna()
        & ~original_end_dates.astype(str)
        .str.strip()
        .isin(["", "0", "٠", "nan"])
    )

    invalid_end_dates = (
        valid_original
        & df["تاريخ انتهاء الخدمة"].isna()
    ).sum()

negative_service = 0

if (
    "تاريخ التعيين" in df.columns
    and "تاريخ انتهاء الخدمة" in df.columns
):
    negative_service = (
        df["تاريخ انتهاء الخدمة"]
        < df["تاريخ التعيين"]
    ).sum()


# ============================================================
# Sidebar Filters
# ============================================================
st.sidebar.header("🔎 الفلاتر")

filtered_df = df.copy()


def multiselect_filter(data, column, label):
    if column not in data.columns:
        return data

    values = sorted(
        data[column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if len(values) == 0:
        return data

    selected = st.sidebar.multiselect(
        label,
        options=values
    )

    if selected:
        data = data[
            data[column]
            .astype(str)
            .isin(selected)
        ]

    return data


filtered_df = multiselect_filter(
    filtered_df,
    "رمز الدائرة",
    "الدائرة"
)

# السنوات
if "سنة انتهاء الخدمة" in filtered_df.columns:

    years = sorted(
        filtered_df["سنة انتهاء الخدمة"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    selected_years = st.sidebar.multiselect(
        "سنة انتهاء الخدمة",
        years
    )

    if selected_years:
        filtered_df = filtered_df[
            filtered_df["سنة انتهاء الخدمة"]
            .isin(selected_years)
        ]

filtered_df = multiselect_filter(
    filtered_df,
    "سبب انتهاء الخدمة",
    "سبب انتهاء الخدمة"
)

filtered_df = multiselect_filter(
    filtered_df,
    "سبب الاستقالة",
    "سبب الاستقالة"
)

filtered_df = multiselect_filter(
    filtered_df,
    "فئة الجنسية",
    "فئة الجنسية"
)

filtered_df = multiselect_filter(
    filtered_df,
    "الجنسية",
    "الجنسية"
)

filtered_df = multiselect_filter(
    filtered_df,
    "الفئة الوظيفية",
    "الفئة الوظيفية"
)

filtered_df = multiselect_filter(
    filtered_df,
    "المجموعة الوظيفية الرئيسية",
    "المجموعة الوظيفية الرئيسية"
)

filtered_df = multiselect_filter(
    filtered_df,
    "المجموعة الوظيفية الفرعية",
    "المجموعة الوظيفية الفرعية"
)

filtered_df = multiselect_filter(
    filtered_df,
    "الدرجة الوظيفية",
    "الدرجة الوظيفية"
)

# ============================================================
# حالة عدم وجود نتائج
# ============================================================
if filtered_df.empty:
    st.warning("لا توجد نتائج مطابقة للفلاتر الحالية.")
    st.stop()


# ============================================================
# اختيار نوع التحليل
# ============================================================
st.divider()

analysis_option = st.selectbox(
    "📌 اختر نوع التحليل",
    [
        "نظرة عامة",
        "تحليل حسب الدائرة",
        "تحليل أسباب انتهاء الخدمة",
        "تحليل الاستقالات",
        "تحليل مدة الخدمة",
        "تحليل الفئات والمجموعات الوظيفية",
        "تحليل الدرجات والمسميات الوظيفية",
        "تحليل الجنسية",
        "التحليل المالي",
        "الاتجاهات الزمنية Trends",
        "الخروج المبكر من الخدمة",
        "جودة البيانات"
    ]
)

# ============================================================
# KPIs عامة
# ============================================================
employee_count = count_unique_employees(filtered_df)

resignation_mask = find_resignation_mask(filtered_df)
resignation_count = count_unique_employees(
    filtered_df[resignation_mask]
)

resignation_pct = (
    resignation_count / employee_count * 100
    if employee_count > 0
    else 0
)

avg_service = (
    filtered_df["مدة الخدمة بالسنوات"].mean()
    if "مدة الخدمة بالسنوات" in filtered_df.columns
    else np.nan
)

median_service = (
    filtered_df["مدة الخدمة بالسنوات"].median()
    if "مدة الخدمة بالسنوات" in filtered_df.columns
    else np.nan
)

total_salary = (
    filtered_df["مجموع الراتب"].sum()
    if "مجموع الراتب" in filtered_df.columns
    else np.nan
)

avg_salary = (
    filtered_df["مجموع الراتب"].mean()
    if "مجموع الراتب" in filtered_df.columns
    else np.nan
)


# ============================================================
# 1. نظرة عامة
# ============================================================
if analysis_option == "نظرة عامة":

    st.subheader("📌 المؤشرات الرئيسية")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "إجمالي المنتهية خدماتهم",
        format_number(employee_count)
    )

    c2.metric(
        "عدد الاستقالات",
        format_number(resignation_count)
    )

    c3.metric(
        "نسبة الاستقالات",
        f"{resignation_pct:.1f}%"
    )

    c4, c5, c6 = st.columns(3)

    c4.metric(
        "متوسط مدة الخدمة",
        (
            f"{avg_service:.2f} سنة"
            if pd.notna(avg_service)
            else "—"
        )
    )

    c5.metric(
        "إجمالي مجموع الرواتب",
        (
            f"{total_salary:,.2f}"
            if pd.notna(total_salary)
            else "—"
        )
    )

    c6.metric(
        "متوسط مجموع الراتب",
        (
            f"{avg_salary:,.2f}"
            if pd.notna(avg_salary)
            else "—"
        )
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### أسباب انتهاء الخدمة")

        reason_table = build_count_table(
            filtered_df,
            "سبب انتهاء الخدمة"
        )

        if not reason_table.empty:
            fig = px.pie(
                reason_table,
                names="سبب انتهاء الخدمة",
                values="العدد",
                hole=0.45
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key="overview_reasons"
            )

    with col2:

        st.markdown("### المنتهية خدماتهم حسب الدائرة")

        dept_table = build_count_table(
            filtered_df,
            "رمز الدائرة"
        )

        if not dept_table.empty:
            fig = px.bar(
                dept_table.sort_values(
                    "العدد",
                    ascending=True
                ),
                x="العدد",
                y="رمز الدائرة",
                orientation="h",
                text="العدد"
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key="overview_departments"
            )

    # Trend
    if "سنة انتهاء الخدمة" in filtered_df.columns:

        st.markdown("### 📈 الاتجاه السنوي")

        trend = (
            filtered_df
            .dropna(subset=["سنة انتهاء الخدمة"])
            .groupby("سنة انتهاء الخدمة")
            .size()
            .reset_index(name="العدد")
            .sort_values("سنة انتهاء الخدمة")
        )

        if not trend.empty:
            fig = px.line(
                trend,
                x="سنة انتهاء الخدمة",
                y="العدد",
                markers=True
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key="overview_trend"
            )


# ============================================================
# 2. تحليل حسب الدائرة
# ============================================================
elif analysis_option == "تحليل حسب الدائرة":

    st.subheader("🏢 تحليل المنتهية خدماتهم حسب الدائرة")

    if "رمز الدائرة" not in filtered_df.columns:
        st.warning("حقل رمز الدائرة غير موجود.")

    else:

        dept = (
            filtered_df
            .groupby("رمز الدائرة", dropna=False)
            .agg(
                عدد_الحالات=(
                    "رمز الدائرة",
                    "size"
                )
            )
            .reset_index()
        )

        # عدد الموظفين الفريدين
        if "الرقم الوظيفي" in filtered_df.columns:
            unique_emp = (
                filtered_df
                .groupby("رمز الدائرة")[
                    "الرقم الوظيفي"
                ]
                .nunique()
                .reset_index(
                    name="عدد المنتهية خدماتهم"
                )
            )

            dept = dept.merge(
                unique_emp,
                on="رمز الدائرة",
                how="left"
            )

        else:
            dept["عدد المنتهية خدماتهم"] = (
                dept["عدد_الحالات"]
            )

        # متوسط مدة الخدمة
        if "مدة الخدمة بالسنوات" in filtered_df.columns:
            temp = (
                filtered_df
                .groupby("رمز الدائرة")[
                    "مدة الخدمة بالسنوات"
                ]
                .mean()
                .round(2)
                .reset_index(
                    name="متوسط مدة الخدمة"
                )
            )

            dept = dept.merge(
                temp,
                on="رمز الدائرة",
                how="left"
            )

        # متوسط الراتب
        if "مجموع الراتب" in filtered_df.columns:
            temp = (
                filtered_df
                .groupby("رمز الدائرة")[
                    "مجموع الراتب"
                ]
                .mean()
                .round(2)
                .reset_index(
                    name="متوسط مجموع الراتب"
                )
            )

            dept = dept.merge(
                temp,
                on="رمز الدائرة",
                how="left"
            )

        # السبب الأكثر شيوعاً
        if "سبب انتهاء الخدمة" in filtered_df.columns:
            top_reason = (
                filtered_df
                .groupby("رمز الدائرة")[
                    "سبب انتهاء الخدمة"
                ]
                .agg(safe_mode)
                .reset_index(
                    name="أكثر سبب انتهاء خدمة"
                )
            )

            dept = dept.merge(
                top_reason,
                on="رمز الدائرة",
                how="left"
            )

        total_dept = dept[
            "عدد المنتهية خدماتهم"
        ].sum()

        dept["النسبة من الإجمالي %"] = (
            dept["عدد المنتهية خدماتهم"]
            / total_dept
            * 100
        ).round(2)

        dept = dept.sort_values(
            "عدد المنتهية خدماتهم",
            ascending=False
        )

        st.dataframe(
            dept,
            use_container_width=True,
            hide_index=True
        )

        fig = px.bar(
            dept.sort_values(
                "عدد المنتهية خدماتهم"
            ),
            x="عدد المنتهية خدماتهم",
            y="رمز الدائرة",
            orientation="h",
            text="عدد المنتهية خدماتهم"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="dept_analysis"
        )

        # الوحدة التنظيمية
        if "الوحدة التنظيمية" in filtered_df.columns:

            st.markdown(
                "### الوحدات التنظيمية الأعلى في انتهاء الخدمة"
            )

            units = build_count_table(
                filtered_df,
                "الوحدة التنظيمية"
            ).head(20)

            st.dataframe(
                units,
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# 3. أسباب انتهاء الخدمة
# ============================================================
elif analysis_option == "تحليل أسباب انتهاء الخدمة":

    st.subheader("📋 تحليل أسباب انتهاء الخدمة")

    reasons = build_count_table(
        filtered_df,
        "سبب انتهاء الخدمة"
    )

    if reasons.empty:
        st.warning("لا توجد بيانات لسبب انتهاء الخدمة.")

    else:

        st.dataframe(
            reasons,
            use_container_width=True,
            hide_index=True
        )

        show_bar_chart(
            reasons,
            "سبب انتهاء الخدمة",
            title="توزيع أسباب انتهاء الخدمة",
            horizontal=True
        )


        # السبب × السنة
        if "سنة انتهاء الخدمة" in filtered_df.columns:

            reason_year = (
                filtered_df
                .dropna(
                    subset=["سنة انتهاء الخدمة"]
                )
                .groupby(
                    [
                        "سنة انتهاء الخدمة",
                        "سبب انتهاء الخدمة"
                    ],
                    dropna=False
                )
                .size()
                .reset_index(name="العدد")
            )

            if not reason_year.empty:

                fig = px.line(
                    reason_year,
                    x="سنة انتهاء الخدمة",
                    y="العدد",
                    color="سبب انتهاء الخدمة",
                    markers=True
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key="reason_year"
                )


# ============================================================
# 4. تحليل الاستقالات
# ============================================================
elif analysis_option == "تحليل الاستقالات":

    st.subheader("🚪 تحليل الاستقالات")

    resignation_df = filtered_df[
        resignation_mask
    ].copy()

    if resignation_df.empty:
        st.info(
            "لا توجد حالات تم التعرف عليها كاستقالة "
            "ضمن الفلاتر الحالية."
        )

    else:

        resignation_employees = (
            count_unique_employees(resignation_df)
        )

        avg_resignation_service = (
            resignation_df[
                "مدة الخدمة بالسنوات"
            ].mean()
            if "مدة الخدمة بالسنوات"
            in resignation_df.columns
            else np.nan
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "عدد الاستقالات",
            format_number(
                resignation_employees
            )
        )

        c2.metric(
            "نسبة الاستقالات من الحالات",
            f"{resignation_pct:.1f}%"
        )

        c3.metric(
            "متوسط الخدمة قبل الاستقالة",
            (
                f"{avg_resignation_service:.2f} سنة"
                if pd.notna(
                    avg_resignation_service
                )
                else "—"
            )
        )

        if "سبب الاستقالة" in resignation_df.columns:

            st.markdown("### أسباب الاستقالة")

            resignation_reasons = build_count_table(
                resignation_df,
                "سبب الاستقالة"
            )

            st.dataframe(
                resignation_reasons,
                use_container_width=True,
                hide_index=True
            )

            show_bar_chart(
                resignation_reasons.head(20),
                "سبب الاستقالة",
                title="أسباب الاستقالة",
                horizontal=True
            )

# ============================================================
# أسباب الاستقالة حسب الدائرة
# ============================================================

if (
    "رمز الدائرة" in resignation_df.columns
    and "سبب الاستقالة" in resignation_df.columns
):

    st.markdown("### 📋 أسباب الاستقالة حسب الدائرة")

    resignation_reason_dept = resignation_df.copy()

    resignation_reason_dept["سبب الاستقالة"] = (
        resignation_reason_dept["سبب الاستقالة"]
        .fillna("غير محدد")
        .astype(str)
        .str.strip()
    )

    if "الرقم الوظيفي" in resignation_reason_dept.columns:

        resignation_reason_dept = (
            resignation_reason_dept
            .groupby(
                ["رمز الدائرة", "سبب الاستقالة"]
            )["الرقم الوظيفي"]
            .nunique()
            .reset_index(name="العدد")
        )

    else:

        resignation_reason_dept = (
            resignation_reason_dept
            .groupby(
                ["رمز الدائرة", "سبب الاستقالة"]
            )
            .size()
            .reset_index(name="العدد")
        )

    resignation_reason_dept = (
        resignation_reason_dept
        .sort_values(
            ["رمز الدائرة", "العدد"],
            ascending=[True, False]
        )
    )

    st.dataframe(
        resignation_reason_dept,
        use_container_width=True,
        hide_index=True
    )

        if (
            "سنة انتهاء الخدمة"
            in resignation_df.columns
        ):

            st.markdown(
                "### اتجاه الاستقالات عبر السنوات"
            )

            resignation_trend = (
                resignation_df
                .dropna(
                    subset=["سنة انتهاء الخدمة"]
                )
                .groupby("سنة انتهاء الخدمة")
                .size()
                .reset_index(name="العدد")
            )

            fig = px.line(
                resignation_trend,
                x="سنة انتهاء الخدمة",
                y="العدد",
                markers=True
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key="resignation_trend"
            )


# ============================================================
# 5. تحليل مدة الخدمة
# ============================================================
elif analysis_option == "تحليل مدة الخدمة":

    st.subheader("⏳ تحليل مدة الخدمة")

    if "مدة الخدمة بالسنوات" not in filtered_df.columns:
        st.warning(
            "لا يمكن حساب مدة الخدمة لعدم توفر "
            "تاريخ التعيين وتاريخ انتهاء الخدمة."
        )

    else:

        service = filtered_df[
            "مدة الخدمة بالسنوات"
        ].dropna()

        if service.empty:
            st.warning(
                "لا توجد مدد خدمة صالحة للتحليل."
            )

        else:

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "متوسط مدة الخدمة",
                f"{service.mean():.2f} سنة"
            )

            c2.metric(
                "الوسيط",
                f"{service.median():.2f} سنة"
            )

            c3.metric(
                "أقل مدة",
                f"{service.min():.2f} سنة"
            )

            c4.metric(
                "أعلى مدة",
                f"{service.max():.2f} سنة"
            )

            service_table = build_count_table(
                filtered_df,
                "فئة مدة الخدمة"
            )

            # ترتيب الفئات
            order = [
                "أقل من سنة",
                "1 - 3 سنوات",
                "3 - 5 سنوات",
                "5 - 10 سنوات",
                "10 - 15 سنة",
                "أكثر من 15 سنة"
            ]

            if not service_table.empty:

                service_table[
                    "فئة مدة الخدمة"
                ] = service_table[
                    "فئة مدة الخدمة"
                ].astype(str)

                service_table["_order"] = (
                    service_table[
                        "فئة مدة الخدمة"
                    ].map(
                        {
                            v: i
                            for i, v in enumerate(
                                order
                            )
                        }
                    )
                )

                service_table = (
                    service_table
                    .sort_values("_order")
                    .drop(columns="_order")
                )

                st.dataframe(
                    service_table,
                    use_container_width=True,
                    hide_index=True
                )

                show_bar_chart(
                    service_table,
                    "فئة مدة الخدمة",
                    title="توزيع الموظفين حسب مدة الخدمة"
                )

            # حسب الدائرة
            if "رمز الدائرة" in filtered_df.columns:

                st.markdown(
                    "### متوسط مدة الخدمة حسب الدائرة"
                )

                service_dept = (
                    filtered_df
                    .groupby("رمز الدائرة")[
                        "مدة الخدمة بالسنوات"
                    ]
                    .agg(
                        [
                            "count",
                            "mean",
                            "median",
                            "min",
                            "max"
                        ]
                    )
                    .round(2)
                    .reset_index()
                )

                service_dept.columns = [
                    "رمز الدائرة",
                    "عدد الحالات",
                    "متوسط مدة الخدمة",
                    "الوسيط",
                    "أقل مدة",
                    "أعلى مدة"
                ]

                st.dataframe(
                    service_dept,
                    use_container_width=True,
                    hide_index=True
                )


# ============================================================
# 6. الفئات والمجموعات الوظيفية
# ============================================================
elif analysis_option == "تحليل الفئات والمجموعات الوظيفية":

    st.subheader(
        "👥 تحليل الفئات والمجموعات الوظيفية"
    )

    job_columns = [
        "الفئة الوظيفية",
        "المجموعة الوظيفية الرئيسية",
        "المجموعة الوظيفية الفرعية"
    ]

    for col in job_columns:

        if col in filtered_df.columns:

            st.markdown(f"### {col}")

            table = build_count_table(
                filtered_df,
                col
            )

            st.dataframe(
                table,
                use_container_width=True,
                hide_index=True
            )

            show_bar_chart(
                table.head(20),
                col,
                title=col,
                horizontal=True
            )


# ============================================================
# 7. الدرجات والمسميات
# ============================================================
elif analysis_option == "تحليل الدرجات والمسميات الوظيفية":

    st.subheader(
        "💼 تحليل الدرجات والمسميات الوظيفية"
    )

    if "الدرجة الوظيفية" in filtered_df.columns:

        st.markdown("### الدرجات الوظيفية")

        grade_table = build_count_table(
            filtered_df,
            "الدرجة الوظيفية"
        )

        st.dataframe(
            grade_table,
            use_container_width=True,
            hide_index=True
        )

        show_bar_chart(
            grade_table,
            "الدرجة الوظيفية",
            title="انتهاء الخدمة حسب الدرجة الوظيفية",
            horizontal=True
        )

    if "المسمى الوظيفي" in filtered_df.columns:

        st.markdown(
            "### أعلى المسميات الوظيفية"
        )

        title_table = build_count_table(
            filtered_df,
            "المسمى الوظيفي"
        )

        st.dataframe(
            title_table,
            use_container_width=True,
            hide_index=True
        )

        show_bar_chart(
            title_table.head(20),
            "المسمى الوظيفي",
            title="أعلى 20 مسمى وظيفي",
            horizontal=True
        )


# ============================================================
# 8. الجنسية
# ============================================================
elif analysis_option == "تحليل الجنسية":

    st.subheader("🌍 تحليل الجنسية")

    if "فئة الجنسية" in filtered_df.columns:

        nationality_category = build_count_table(
            filtered_df,
            "فئة الجنسية"
        )

        st.markdown("### فئة الجنسية")

        st.dataframe(
            nationality_category,
            use_container_width=True,
            hide_index=True
        )

        fig = px.pie(
            nationality_category,
            names="فئة الجنسية",
            values="العدد",
            hole=0.45
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="nationality_category"
        )

    if "الجنسية" in filtered_df.columns:

        st.markdown("### الجنسية")

        nationality = build_count_table(
            filtered_df,
            "الجنسية"
        )

        st.dataframe(
            nationality,
            use_container_width=True,
            hide_index=True
        )

        show_bar_chart(
            nationality.head(20),
            "الجنسية",
            title="أعلى الجنسيات",
            horizontal=True
        )


# ============================================================
# 9. التحليل المالي
# ============================================================
elif analysis_option == "التحليل المالي":

    st.subheader("💰 التحليل المالي")

    salary_available = (
        "مجموع الراتب" in filtered_df.columns
    )

    basic_available = (
        "الراتب الاساسي" in filtered_df.columns
    )

    if not salary_available and not basic_available:
        st.warning("حقول الرواتب غير موجودة.")

    else:

        c1, c2, c3, c4 = st.columns(4)

        if salary_available:

            c1.metric(
                "إجمالي مجموع الرواتب",
                f"{filtered_df['مجموع الراتب'].sum():,.2f}"
            )

            c2.metric(
                "متوسط مجموع الراتب",
                f"{filtered_df['مجموع الراتب'].mean():,.2f}"
            )

        if basic_available:

            c3.metric(
                "إجمالي الراتب الأساسي",
                f"{filtered_df['الراتب الاساسي'].sum():,.2f}"
            )

            c4.metric(
                "متوسط الراتب الأساسي",
                f"{filtered_df['الراتب الاساسي'].mean():,.2f}"
            )

        # حسب الدائرة
        if (
            "رمز الدائرة" in filtered_df.columns
            and salary_available
        ):

            st.markdown(
                "### الرواتب حسب الدائرة"
            )

            financial_dept = (
                filtered_df
                .groupby("رمز الدائرة")
                .agg(
                    عدد_الحالات=(
                        "رمز الدائرة",
                        "size"
                    ),
                    إجمالي_مجموع_الرواتب=(
                        "مجموع الراتب",
                        "sum"
                    ),
                    متوسط_مجموع_الراتب=(
                        "مجموع الراتب",
                        "mean"
                    )
                )
                .round(2)
                .reset_index()
            )

            if basic_available:

                basic = (
                    filtered_df
                    .groupby("رمز الدائرة")[
                        "الراتب الاساسي"
                    ]
                    .agg(["sum", "mean"])
                    .round(2)
                    .reset_index()
                )

                basic.columns = [
                    "رمز الدائرة",
                    "إجمالي الراتب الأساسي",
                    "متوسط الراتب الأساسي"
                ]

                financial_dept = (
                    financial_dept.merge(
                        basic,
                        on="رمز الدائرة",
                        how="left"
                    )
                )

            st.dataframe(
                financial_dept,
                use_container_width=True,
                hide_index=True
            )

        # حسب السبب
        if (
            "سبب انتهاء الخدمة"
            in filtered_df.columns
            and salary_available
        ):

            st.markdown(
                "### الرواتب حسب سبب انتهاء الخدمة"
            )

            salary_reason = (
                filtered_df
                .groupby(
                    "سبب انتهاء الخدمة",
                    dropna=False
                )["مجموع الراتب"]
                .agg(
                    [
                        "count",
                        "sum",
                        "mean",
                        "median"
                    ]
                )
                .round(2)
                .reset_index()
            )

            salary_reason.columns = [
                "سبب انتهاء الخدمة",
                "عدد الحالات",
                "إجمالي مجموع الرواتب",
                "متوسط مجموع الراتب",
                "وسيط مجموع الراتب"
            ]

            st.dataframe(
                salary_reason,
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# 10. Trends
# ============================================================
elif analysis_option == "الاتجاهات الزمنية Trends":

    st.subheader("📈 الاتجاهات الزمنية")

    if "سنة انتهاء الخدمة" not in filtered_df.columns:
        st.warning(
            "لا يمكن إجراء التحليل الزمني بدون "
            "تاريخ انتهاء الخدمة."
        )

    else:

        # سنوي
        yearly = (
            filtered_df
            .dropna(subset=["سنة انتهاء الخدمة"])
            .groupby("سنة انتهاء الخدمة")
            .size()
            .reset_index(name="العدد")
            .sort_values("سنة انتهاء الخدمة")
        )

        st.markdown("### الاتجاه السنوي")

        if not yearly.empty:

            fig = px.line(
                yearly,
                x="سنة انتهاء الخدمة",
                y="العدد",
                markers=True,
                text="العدد"
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key="yearly_trend"
            )

        # شهري
        if "شهر انتهاء الخدمة" in filtered_df.columns:

            monthly = (
                filtered_df
                .dropna(
                    subset=["تاريخ انتهاء الخدمة"]
                )
                .groupby("شهر انتهاء الخدمة")
                .size()
                .reset_index(name="العدد")
                .sort_values(
                    "شهر انتهاء الخدمة"
                )
            )

            st.markdown("### الاتجاه الشهري")

            if not monthly.empty:

                fig = px.line(
                    monthly,
                    x="شهر انتهاء الخدمة",
                    y="العدد",
                    markers=True
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key="monthly_trend"
                )

        # سنة × دائرة
        if "رمز الدائرة" in filtered_df.columns:

            st.markdown(
                "### الاتجاه السنوي حسب الدائرة"
            )

            yearly_dept = (
                filtered_df
                .dropna(
                    subset=["سنة انتهاء الخدمة"]
                )
                .groupby(
                    [
                        "سنة انتهاء الخدمة",
                        "رمز الدائرة"
                    ]
                )
                .size()
                .reset_index(name="العدد")
            )

            fig = px.line(
                yearly_dept,
                x="سنة انتهاء الخدمة",
                y="العدد",
                color="رمز الدائرة",
                markers=True
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key="yearly_dept_trend"
            )

        # سنة × السبب
        if (
            "سبب انتهاء الخدمة"
            in filtered_df.columns
        ):

            st.markdown(
                "### الاتجاه السنوي حسب سبب انتهاء الخدمة"
            )

            yearly_reason = (
                filtered_df
                .dropna(
                    subset=["سنة انتهاء الخدمة"]
                )
                .groupby(
                    [
                        "سنة انتهاء الخدمة",
                        "سبب انتهاء الخدمة"
                    ],
                    dropna=False
                )
                .size()
                .reset_index(name="العدد")
            )

            fig = px.line(
                yearly_reason,
                x="سنة انتهاء الخدمة",
                y="العدد",
                color="سبب انتهاء الخدمة",
                markers=True
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key="yearly_reason_trend"
            )


# ============================================================
# 11. الخروج المبكر
# ============================================================
elif analysis_option == "الخروج المبكر من الخدمة":

    st.subheader("⚠️ تحليل الخروج المبكر من الخدمة")

    if "مدة الخدمة بالسنوات" not in filtered_df.columns:
        st.warning(
            "لا يمكن حساب الخروج المبكر بدون مدة الخدمة."
        )

    else:

        valid_service_df = filtered_df[
            filtered_df[
                "مدة الخدمة بالسنوات"
            ].notna()
        ].copy()

        if valid_service_df.empty:
            st.warning(
                "لا توجد مدد خدمة صالحة."
            )

        else:

            less_1 = valid_service_df[
                valid_service_df[
                    "مدة الخدمة بالسنوات"
                ] < 1
            ]

            less_3 = valid_service_df[
                valid_service_df[
                    "مدة الخدمة بالسنوات"
                ] < 3
            ]

            total_valid = count_unique_employees(
                valid_service_df
            )

            less_1_count = count_unique_employees(
                less_1
            )

            less_3_count = count_unique_employees(
                less_3
            )

            less_1_pct = (
                less_1_count / total_valid * 100
                if total_valid
                else 0
            )

            less_3_pct = (
                less_3_count / total_valid * 100
                if total_valid
                else 0
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "خروج خلال أول سنة",
                format_number(less_1_count)
            )

            c2.metric(
                "نسبة الخروج خلال أول سنة",
                f"{less_1_pct:.1f}%"
            )

            c3.metric(
                "خروج خلال أول 3 سنوات",
                format_number(less_3_count)
            )

            c4.metric(
                "نسبة الخروج خلال أول 3 سنوات",
                f"{less_3_pct:.1f}%"
            )

            # تحليل أقل من 3 سنوات
            st.markdown(
                "### تفاصيل حالات الخروج خلال أول 3 سنوات"
            )

            if "رمز الدائرة" in less_3.columns:

                early_dept = build_count_table(
                    less_3,
                    "رمز الدائرة"
                )

                st.markdown(
                    "#### حسب الدائرة"
                )

                st.dataframe(
                    early_dept,
                    use_container_width=True,
                    hide_index=True
                )

                show_bar_chart(
                    early_dept,
                    "رمز الدائرة",
                    title="الخروج المبكر حسب الدائرة",
                    horizontal=True
                )

            if (
                "سبب انتهاء الخدمة"
                in less_3.columns
            ):

                early_reason = build_count_table(
                    less_3,
                    "سبب انتهاء الخدمة"
                )

                st.markdown(
                    "#### حسب سبب انتهاء الخدمة"
                )

                st.dataframe(
                    early_reason,
                    use_container_width=True,
                    hide_index=True
                )

            if "المسمى الوظيفي" in less_3.columns:

                early_job = build_count_table(
                    less_3,
                    "المسمى الوظيفي"
                ).head(20)

                st.markdown(
                    "#### أعلى المسميات في الخروج المبكر"
                )

                st.dataframe(
                    early_job,
                    use_container_width=True,
                    hide_index=True
                )

            # الاستقالات المبكرة
            early_resignation = less_3[
                find_resignation_mask(less_3)
            ]

            early_resignation_count = (
                count_unique_employees(
                    early_resignation
                )
            )

            st.metric(
                "الاستقالات خلال أول 3 سنوات",
                format_number(
                    early_resignation_count
                )
            )


# ============================================================
# 12. جودة البيانات
# ============================================================
elif analysis_option == "جودة البيانات":

    st.subheader("🧹 جودة البيانات بعد التنظيف")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "عدد الصفوف الأصلية",
        format_number(original_row_count)
    )

    c2.metric(
        "تواريخ تعيين غير مقروءة",
        format_number(invalid_hire_dates)
    )

    c3.metric(
        "تواريخ انتهاء غير مقروءة",
        format_number(invalid_end_dates)
    )

    c4.metric(
        "انتهاء قبل تاريخ التعيين",
        format_number(negative_service)
    )

    quality_rows = []

    for col in EXPECTED_COLUMNS:

        if col in df.columns:

            missing = df[col].isna().sum()

            quality_rows.append(
                {
                    "الحقل": col,
                    "عدد السجلات": len(df),
                    "القيم المفقودة": missing,
                    "نسبة الاكتمال %": round(
                        (
                            1 - missing / len(df)
                        ) * 100,
                        2
                    )
                    if len(df) > 0
                    else 0
                }
            )

    quality_table = pd.DataFrame(
        quality_rows
    )

    st.dataframe(
        quality_table,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### عينة من البيانات بعد التنظيف")

    preview_cols = [
        c for c in [
            "الرقم الوظيفي",
            "اسم الموظف",
            "تاريخ التعيين",
            "تاريخ انتهاء الخدمة",
            "مدة الخدمة بالسنوات",
            "مجموع الراتب",
            "الراتب الاساسي"
        ]
        if c in df.columns
    ]

    st.dataframe(
        df[preview_cols].head(50),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# تصدير النتائج إلى Excel
# ============================================================
st.divider()
st.subheader("📥 تصدير النتائج")

export_sheets = {
    "البيانات بعد التنظيف": filtered_df
}

# إضافة جداول أساسية
if "رمز الدائرة" in filtered_df.columns:
    export_sheets["حسب الدائرة"] = (
        build_count_table(
            filtered_df,
            "رمز الدائرة"
        )
    )

if "سبب انتهاء الخدمة" in filtered_df.columns:
    export_sheets["أسباب انتهاء الخدمة"] = (
        build_count_table(
            filtered_df,
            "سبب انتهاء الخدمة"
        )
    )

if "سبب الاستقالة" in filtered_df.columns:
    resignation_export = filtered_df[
        find_resignation_mask(filtered_df)
    ]

    export_sheets["أسباب الاستقالة"] = (
        build_count_table(
            resignation_export,
            "سبب الاستقالة"
        )
    )

if "فئة مدة الخدمة" in filtered_df.columns:
    export_sheets["مدة الخدمة"] = (
        build_count_table(
            filtered_df,
            "فئة مدة الخدمة"
        )
    )

if "سنة انتهاء الخدمة" in filtered_df.columns:
    export_sheets["الاتجاه السنوي"] = (
        filtered_df
        .dropna(subset=["سنة انتهاء الخدمة"])
        .groupby("سنة انتهاء الخدمة")
        .size()
        .reset_index(name="العدد")
    )

# جدول مؤشرات عامة
summary_data = {
    "المؤشر": [
        "إجمالي المنتهية خدماتهم",
        "عدد الاستقالات",
        "نسبة الاستقالات %",
        "متوسط مدة الخدمة",
        "وسيط مدة الخدمة",
        "إجمالي مجموع الرواتب",
        "متوسط مجموع الراتب"
    ],
    "القيمة": [
        employee_count,
        resignation_count,
        round(resignation_pct, 2),
        round(avg_service, 2)
        if pd.notna(avg_service)
        else np.nan,
        round(median_service, 2)
        if pd.notna(median_service)
        else np.nan,
        round(total_salary, 2)
        if pd.notna(total_salary)
        else np.nan,
        round(avg_salary, 2)
        if pd.notna(avg_salary)
        else np.nan
    ]
}

export_sheets[
    "المؤشرات العامة"
] = pd.DataFrame(summary_data)

try:

    excel_output = create_excel_download(
        export_sheets
    )

    st.download_button(
        label="⬇️ تحميل نتائج التحليل Excel",
        data=excel_output,
        file_name="تحليل_المنتهية_خدماتهم.xlsx",
        mime=(
            "application/vnd.openxmlformats-"
            "officedocument.spreadsheetml.sheet"
        )
    )

except Exception as e:
    st.error(
        f"تعذر إنشاء ملف Excel: {e}"
    )


# ============================================================
# عرض البيانات التفصيلية
# ============================================================
with st.expander(
    "🔍 عرض البيانات التفصيلية بعد التنظيف"
):

    st.write(
        f"عدد السجلات بعد تطبيق الفلاتر: "
        f"{len(filtered_df):,}"
    )

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )
