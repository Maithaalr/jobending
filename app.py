import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
from io import BytesIO


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="تحليل المنتهية خدماتهم",
    page_icon="📊",
    layout="wide"
)

st.title("📊 لوحة تحليل المنتهية خدماتهم")

st.caption(
    "تحليل حالات انتهاء الخدمة والاستقالات ومدة الخدمة "
    "والرواتب والاتجاهات الزمنية"
)


# ============================================================
# CONSTANTS
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

ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
ENGLISH_DIGITS = "0123456789"

TRANSLATION_TABLE = str.maketrans(
    ARABIC_DIGITS + "٫٬",
    ENGLISH_DIGITS + ".,"
)

HIDDEN_CHARS_PATTERN = (
    r"[\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_column_name(col):

    col = str(col)

    col = re.sub(
        HIDDEN_CHARS_PATTERN,
        "",
        col
    )

    col = col.replace("\xa0", " ")

    col = re.sub(
        r"\s+",
        " ",
        col
    )

    return col.strip()


def normalize_text(value):

    if pd.isna(value):
        return np.nan

    value = str(value)

    value = re.sub(
        HIDDEN_CHARS_PATTERN,
        "",
        value
    )

    value = value.replace(
        "\xa0",
        " "
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    value = value.strip()

    if value.lower() in [
        "",
        "nan",
        "none",
        "null"
    ]:
        return np.nan

    return value


def arabic_to_english_numbers(value):

    if pd.isna(value):
        return value

    return str(value).translate(
        TRANSLATION_TABLE
    )


def clean_date(value):

    if pd.isna(value):
        return pd.NaT

    value = arabic_to_english_numbers(
        value
    )

    value = re.sub(
        HIDDEN_CHARS_PATTERN,
        "",
        str(value)
    )

    value = value.replace(
        "\xa0",
        " "
    ).strip()

    if value.lower() in [
        "",
        "0",
        "nan",
        "none",
        "null",
        "-",
        "--"
    ]:
        return pd.NaT

    # First attempt: DD/MM/YYYY
    result = pd.to_datetime(
        value,
        format="%d/%m/%Y",
        errors="coerce"
    )

    # Second attempt
    if pd.isna(result):

        result = pd.to_datetime(
            value,
            dayfirst=True,
            errors="coerce"
        )

    return result


def clean_salary(value):

    if pd.isna(value):
        return np.nan

    value = arabic_to_english_numbers(
        value
    )

    value = re.sub(
        HIDDEN_CHARS_PATTERN,
        "",
        str(value)
    )

    value = value.replace(
        "\xa0",
        ""
    )

    value = value.replace(
        " ",
        ""
    )

    value = value.replace(
        ",",
        ""
    )

    value = re.sub(
        r"[^0-9.\-]",
        "",
        value
    )

    if value in [
        "",
        ".",
        "-",
        "-."
    ]:
        return np.nan

    try:
        return float(value)

    except ValueError:
        return np.nan


def format_number(
    value,
    decimals=0
):

    if pd.isna(value):
        return "—"

    return f"{value:,.{decimals}f}"


def safe_mode(series):

    series = series.dropna()

    if series.empty:
        return "—"

    mode_values = series.mode()

    if mode_values.empty:
        return "—"

    return str(
        mode_values.iloc[0]
    )


def count_unique_employees(data):

    if "الرقم الوظيفي" in data.columns:

        values = data[
            "الرقم الوظيفي"
        ].dropna()

        if not values.empty:
            return values.nunique()

    return len(data)


def find_resignation_mask(data):

    if "سبب انتهاء الخدمة" not in data.columns:

        return pd.Series(
            False,
            index=data.index
        )

    values = (
        data["سبب انتهاء الخدمة"]
        .fillna("")
        .astype(str)
    )

    return values.str.contains(
        "استقال",
        case=False,
        regex=False
    )


def calculate_service_fields(data):

    data = data.copy()

    required = [
        "تاريخ التعيين",
        "تاريخ انتهاء الخدمة"
    ]

    if all(
        col in data.columns
        for col in required
    ):

        data["مدة الخدمة بالأيام"] = (
            data["تاريخ انتهاء الخدمة"]
            -
            data["تاريخ التعيين"]
        ).dt.days

        # Invalid negative service
        data.loc[
            data["مدة الخدمة بالأيام"] < 0,
            "مدة الخدمة بالأيام"
        ] = np.nan

        data["مدة الخدمة بالسنوات"] = (
            data["مدة الخدمة بالأيام"]
            / 365.25
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

    data = data.copy()

    if "تاريخ انتهاء الخدمة" in data.columns:

        data["سنة انتهاء الخدمة"] = (
            data[
                "تاريخ انتهاء الخدمة"
            ].dt.year
        )

        data["رقم شهر انتهاء الخدمة"] = (
            data[
                "تاريخ انتهاء الخدمة"
            ].dt.month
        )

        data["شهر انتهاء الخدمة"] = (
            data[
                "تاريخ انتهاء الخدمة"
            ]
            .dt.to_period("M")
            .astype(str)
        )

    return data


def build_count_table(
    data,
    column
):

    if column not in data.columns:
        return pd.DataFrame()

    values = (
        data[column]
        .fillna("غير محدد")
        .astype(str)
        .str.strip()
    )

    table = (
        values
        .value_counts(
            dropna=False
        )
        .rename_axis(column)
        .reset_index(
            name="العدد"
        )
    )

    total = table[
        "العدد"
    ].sum()

    if total > 0:

        table["النسبة %"] = (
            table["العدد"]
            / total
            * 100
        ).round(2)

    else:

        table["النسبة %"] = 0

    return table


def create_excel_download(
    sheets
):

    output = BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        for sheet_name, table in sheets.items():

            if table is None:
                continue

            if not isinstance(
                table,
                pd.DataFrame
            ):
                continue

            safe_name = str(
                sheet_name
            )[:31]

            export_df = (
                table.copy()
            )

            for col in export_df.columns:

                if pd.api.types.is_datetime64_any_dtype(
                    export_df[col]
                ):

                    export_df[col] = (
                        export_df[col]
                        .dt.strftime(
                            "%d/%m/%Y"
                        )
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
    horizontal=False,
    key=None
):

    if table.empty:

        st.info(
            "لا توجد بيانات كافية لعرض الرسم."
        )

        return

    chart_data = table.copy()

    if horizontal:

        chart_data = (
            chart_data
            .sort_values(
                value,
                ascending=True
            )
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

    if key is None:

        key = (
            f"chart_"
            f"{category}_"
            f"{value}_"
            f"{title}"
        )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key=key
    )


def multiselect_filter(
    data,
    column,
    label,
    key
):

    if column not in data.columns:
        return data

    values = sorted(
        data[column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if not values:
        return data

    selected = (
        st.sidebar.multiselect(
            label,
            options=values,
            key=key
        )
    )

    if selected:

        data = data[
            data[column]
            .astype(str)
            .isin(selected)
        ]

    return data


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📂 ارفعي تقرير المنتهية خدماتهم",
    type=[
        "xlsx",
        "xls"
    ]
)

if uploaded_file is None:

    st.info(
        "ارفعي ملف Excel لبدء التحليل."
    )

    st.stop()


# ============================================================
# READ EXCEL
# ============================================================

try:

    excel_file = pd.ExcelFile(
        uploaded_file
    )

except Exception as e:

    st.error(
        f"تعذر قراءة ملف Excel: {e}"
    )

    st.stop()


if len(
    excel_file.sheet_names
) > 1:

    selected_sheet = st.selectbox(
        "اختر ورقة البيانات",
        excel_file.sheet_names
    )

else:

    selected_sheet = (
        excel_file.sheet_names[0]
    )


try:

    df_raw = pd.read_excel(
        uploaded_file,
        sheet_name=selected_sheet
    )

except Exception as e:

    st.error(
        f"حدث خطأ أثناء قراءة البيانات: {e}"
    )

    st.stop()


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df_raw.columns = [
    normalize_column_name(c)
    for c in df_raw.columns
]

df_raw = (
    df_raw
    .dropna(
        how="all"
    )
    .copy()
)

original_row_count = len(
    df_raw
)


# ============================================================
# COLUMN ALIASES
# ============================================================

COLUMN_ALIASES = {
    "الراتب الأساسي":
        "الراتب الاساسي",

    "الراتب الاساسي ":
        "الراتب الاساسي",

    "رمز الدائرة ":
        "رمز الدائرة",

    "الفئة الوظيفية ":
        "الفئة الوظيفية",

    "المجموعة الوظيفية الرئيسية ":
        "المجموعة الوظيفية الرئيسية",

    "المجموعة الوظيفية الفرعية ":
        "المجموعة الوظيفية الفرعية"
}

rename_dict = {}

for old, new in (
    COLUMN_ALIASES.items()
):

    if (
        old in df_raw.columns
        and
        new not in df_raw.columns
    ):

        rename_dict[
            old
        ] = new


df_raw = df_raw.rename(
    columns=rename_dict
)


# ============================================================
# MISSING COLUMNS
# ============================================================

missing_columns = [
    col
    for col in EXPECTED_COLUMNS
    if col not in df_raw.columns
]

if missing_columns:

    with st.expander(
        "⚠️ حقول غير موجودة في الملف"
    ):

        st.write(
            "سيستمر التحليل بالحقول "
            "الموجودة."
        )

        st.write(
            missing_columns
        )


# ============================================================
# DATA CLEANING
# ============================================================

df = df_raw.copy()

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
    "المجموعة الوظيفية الفرعية"
]

for col in TEXT_COLUMNS:

    if col in df.columns:

        df[col] = (
            df[col]
            .apply(
                normalize_text
            )
        )


# ============================================================
# DATES
# ============================================================

if "تاريخ التعيين" in df.columns:

    original_hire_dates = (
        df[
            "تاريخ التعيين"
        ].copy()
    )

    df["تاريخ التعيين"] = (
        df[
            "تاريخ التعيين"
        ]
        .apply(
            clean_date
        )
    )

else:

    original_hire_dates = (
        pd.Series(
            dtype="object"
        )
    )


if "تاريخ انتهاء الخدمة" in df.columns:

    original_end_dates = (
        df[
            "تاريخ انتهاء الخدمة"
        ].copy()
    )

    df["تاريخ انتهاء الخدمة"] = (
        df[
            "تاريخ انتهاء الخدمة"
        ]
        .apply(
            clean_date
        )
    )

else:

    original_end_dates = (
        pd.Series(
            dtype="object"
        )
    )


# ============================================================
# SALARIES
# ============================================================

for salary_col in [
    "مجموع الراتب",
    "الراتب الاساسي"
]:

    if salary_col in df.columns:

        df[salary_col] = (
            df[salary_col]
            .apply(
                clean_salary
            )
        )


# ============================================================
# CALCULATED FIELDS
# ============================================================

df = calculate_service_fields(
    df
)

df = add_time_fields(
    df
)


# ============================================================
# DATA QUALITY COUNTS
# ============================================================

invalid_hire_dates = 0
invalid_end_dates = 0
negative_service = 0


if "تاريخ التعيين" in df.columns:

    valid_original_hire = (
        original_hire_dates
        .notna()
    )

    invalid_hire_dates = (
        valid_original_hire
        &
        df[
            "تاريخ التعيين"
        ].isna()
    ).sum()


if "تاريخ انتهاء الخدمة" in df.columns:

    valid_original_end = (
        original_end_dates
        .notna()
    )

    invalid_end_dates = (
        valid_original_end
        &
        df[
            "تاريخ انتهاء الخدمة"
        ].isna()
    ).sum()


if (
    "تاريخ التعيين" in df.columns
    and
    "تاريخ انتهاء الخدمة" in df.columns
):

    negative_service = (
        df[
            "تاريخ انتهاء الخدمة"
        ]
        <
        df[
            "تاريخ التعيين"
        ]
    ).sum()


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header(
    "🔎 الفلاتر"
)

filtered_df = df.copy()


filtered_df = multiselect_filter(
    filtered_df,
    "رمز الدائرة",
    "الدائرة",
    "filter_department"
)


if (
    "سنة انتهاء الخدمة"
    in filtered_df.columns
):

    years = sorted(
        filtered_df[
            "سنة انتهاء الخدمة"
        ]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    selected_years = (
        st.sidebar.multiselect(
            "سنة انتهاء الخدمة",
            years,
            key="filter_year"
        )
    )

    if selected_years:

        filtered_df = (
            filtered_df[
                filtered_df[
                    "سنة انتهاء الخدمة"
                ].isin(
                    selected_years
                )
            ]
        )


filtered_df = multiselect_filter(
    filtered_df,
    "سبب انتهاء الخدمة",
    "سبب انتهاء الخدمة",
    "filter_end_reason"
)

filtered_df = multiselect_filter(
    filtered_df,
    "سبب الاستقالة",
    "سبب الاستقالة",
    "filter_resignation_reason"
)

filtered_df = multiselect_filter(
    filtered_df,
    "فئة الجنسية",
    "فئة الجنسية",
    "filter_nationality_category"
)

filtered_df = multiselect_filter(
    filtered_df,
    "الجنسية",
    "الجنسية",
    "filter_nationality"
)

filtered_df = multiselect_filter(
    filtered_df,
    "الفئة الوظيفية",
    "الفئة الوظيفية",
    "filter_job_category"
)

filtered_df = multiselect_filter(
    filtered_df,
    "المجموعة الوظيفية الرئيسية",
    "المجموعة الوظيفية الرئيسية",
    "filter_main_group"
)

filtered_df = multiselect_filter(
    filtered_df,
    "المجموعة الوظيفية الفرعية",
    "المجموعة الوظيفية الفرعية",
    "filter_sub_group"
)

filtered_df = multiselect_filter(
    filtered_df,
    "الدرجة الوظيفية",
    "الدرجة الوظيفية",
    "filter_grade"
)


if filtered_df.empty:

    st.warning(
        "لا توجد نتائج مطابقة للفلاتر الحالية."
    )

    st.stop()


# ============================================================
# ANALYSIS OPTION
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
# GENERAL CALCULATIONS
# ============================================================

employee_count = (
    count_unique_employees(
        filtered_df
    )
)

resignation_mask = (
    find_resignation_mask(
        filtered_df
    )
)

resignation_df = (
    filtered_df[
        resignation_mask
    ].copy()
)

resignation_count = (
    count_unique_employees(
        resignation_df
    )
)

if employee_count > 0:

    resignation_pct = (
        resignation_count
        / employee_count
        * 100
    )

else:

    resignation_pct = 0


if (
    "مدة الخدمة بالسنوات"
    in filtered_df.columns
):

    avg_service = (
        filtered_df[
            "مدة الخدمة بالسنوات"
        ].mean()
    )

    median_service = (
        filtered_df[
            "مدة الخدمة بالسنوات"
        ].median()
    )

else:

    avg_service = np.nan
    median_service = np.nan


if (
    "مجموع الراتب"
    in filtered_df.columns
):

    total_salary = (
        filtered_df[
            "مجموع الراتب"
        ].sum()
    )

    avg_salary = (
        filtered_df[
            "مجموع الراتب"
        ].mean()
    )

else:

    total_salary = np.nan
    avg_salary = np.nan


# ============================================================
# OPTION 1 - OVERVIEW
# ============================================================

if analysis_option == "نظرة عامة":

    st.subheader(
        "📌 المؤشرات الرئيسية"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "إجمالي المنتهية خدماتهم",
        format_number(
            employee_count
        )
    )

    c2.metric(
        "عدد الاستقالات",
        format_number(
            resignation_count
        )
    )

    c3.metric(
        "نسبة الاستقالات",
        f"{resignation_pct:.1f}%"
    )

    c4, c5, c6 = st.columns(3)

    if pd.notna(
        avg_service
    ):

        c4.metric(
            "متوسط مدة الخدمة",
            f"{avg_service:.2f} سنة"
        )

    else:

        c4.metric(
            "متوسط مدة الخدمة",
            "—"
        )

    if pd.notna(
        total_salary
    ):

        c5.metric(
            "إجمالي مجموع الرواتب",
            f"{total_salary:,.2f}"
        )

    else:

        c5.metric(
            "إجمالي مجموع الرواتب",
            "—"
        )

    if pd.notna(
        avg_salary
    ):

        c6.metric(
            "متوسط مجموع الراتب",
            f"{avg_salary:,.2f}"
        )

    else:

        c6.metric(
            "متوسط مجموع الراتب",
            "—"
        )

    st.divider()

    # --------------------------------------------------------
    # END REASONS TABLE
    # --------------------------------------------------------

    st.markdown(
        "### 📋 أسباب انتهاء الخدمة"
    )

    reasons_table = (
        build_count_table(
            filtered_df,
            "سبب انتهاء الخدمة"
        )
    )

    if not reasons_table.empty:

        total_end_cases = (
            reasons_table[
                "العدد"
            ].sum()
        )

        st.metric(
            "إجمالي حالات انتهاء الخدمة",
            f"{total_end_cases:,}"
        )

        st.dataframe(
            reasons_table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "سبب انتهاء الخدمة":
                    st.column_config.TextColumn(
                        "سبب انتهاء الخدمة"
                    ),
                "العدد":
                    st.column_config.NumberColumn(
                        "العدد",
                        format="%d"
                    ),
                "النسبة %":
                    st.column_config.NumberColumn(
                        "النسبة %",
                        format="%.2f%%"
                    )
            }
        )

    # --------------------------------------------------------
    # RESIGNATIONS BY DEPARTMENT TABLE
    # --------------------------------------------------------

    st.markdown(
        "### 🚪 عدد المستقيلين حسب الدائرة"
    )

    if (
        not resignation_df.empty
        and
        "رمز الدائرة"
        in resignation_df.columns
    ):

        if (
            "الرقم الوظيفي"
            in resignation_df.columns
        ):

            resignation_dept = (
                resignation_df
                .groupby(
                    "رمز الدائرة"
                )[
                    "الرقم الوظيفي"
                ]
                .nunique()
                .reset_index(
                    name="عدد المستقيلين"
                )
            )

        else:

            resignation_dept = (
                resignation_df
                .groupby(
                    "رمز الدائرة"
                )
                .size()
                .reset_index(
                    name="عدد المستقيلين"
                )
            )

        resignation_dept = (
            resignation_dept
            .sort_values(
                "عدد المستقيلين",
                ascending=False
            )
            .reset_index(
                drop=True
            )
        )

        total_resignations = (
            resignation_dept[
                "عدد المستقيلين"
            ].sum()
        )

        resignation_dept[
            "النسبة من إجمالي الاستقالات %"
        ] = (
            resignation_dept[
                "عدد المستقيلين"
            ]
            / total_resignations
            * 100
        ).round(2)

        resignation_dept.insert(
            0,
            "الترتيب",
            range(
                1,
                len(
                    resignation_dept
                ) + 1
            )
        )

        st.dataframe(
            resignation_dept,
            use_container_width=True,
            hide_index=True,
            column_config={
                "الترتيب":
                    st.column_config.NumberColumn(
                        "الترتيب",
                        format="%d"
                    ),
                "رمز الدائرة":
                    st.column_config.TextColumn(
                        "الدائرة"
                    ),
                "عدد المستقيلين":
                    st.column_config.NumberColumn(
                        "عدد المستقيلين",
                        format="%d"
                    ),
                "النسبة من إجمالي الاستقالات %":
                    st.column_config.NumberColumn(
                        "النسبة من إجمالي الاستقالات",
                        format="%.2f%%"
                    )
            }
        )

    else:

        st.info(
            "لا توجد استقالات ضمن البيانات الحالية."
        )

    # --------------------------------------------------------
    # DEPARTMENT TABLE
    # --------------------------------------------------------

    st.markdown(
        "### 🏢 المنتهية خدماتهم حسب الدائرة"
    )

    dept_table = (
        build_count_table(
            filtered_df,
            "رمز الدائرة"
        )
    )

    if not dept_table.empty:

        st.dataframe(
            dept_table,
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # YEARLY TREND
    # --------------------------------------------------------

    if (
        "سنة انتهاء الخدمة"
        in filtered_df.columns
    ):

        st.markdown(
            "### 📈 الاتجاه السنوي"
        )

        yearly = (
            filtered_df
            .dropna(
                subset=[
                    "سنة انتهاء الخدمة"
                ]
            )
            .groupby(
                "سنة انتهاء الخدمة"
            )
            .size()
            .reset_index(
                name="العدد"
            )
            .sort_values(
                "سنة انتهاء الخدمة"
            )
        )

        if not yearly.empty:

            yearly[
                "سنة انتهاء الخدمة"
            ] = (
                yearly[
                    "سنة انتهاء الخدمة"
                ]
                .astype(int)
            )

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
                key="overview_yearly"
            )


# ============================================================
# OPTION 2 - DEPARTMENT
# ============================================================

elif analysis_option == "تحليل حسب الدائرة":

    st.subheader(
        "🏢 تحليل حسب الدائرة"
    )

    if (
        "رمز الدائرة"
        not in filtered_df.columns
    ):

        st.warning(
            "حقل رمز الدائرة غير موجود."
        )

    else:

        if (
            "الرقم الوظيفي"
            in filtered_df.columns
        ):

            dept_analysis = (
                filtered_df
                .groupby(
                    "رمز الدائرة"
                )[
                    "الرقم الوظيفي"
                ]
                .nunique()
                .reset_index(
                    name="عدد المنتهية خدماتهم"
                )
            )

        else:

            dept_analysis = (
                filtered_df
                .groupby(
                    "رمز الدائرة"
                )
                .size()
                .reset_index(
                    name="عدد المنتهية خدماتهم"
                )
            )

        # Average service
        if (
            "مدة الخدمة بالسنوات"
            in filtered_df.columns
        ):

            service_dept = (
                filtered_df
                .groupby(
                    "رمز الدائرة"
                )[
                    "مدة الخدمة بالسنوات"
                ]
                .mean()
                .round(2)
                .reset_index(
                    name="متوسط مدة الخدمة"
                )
            )

            dept_analysis = (
                dept_analysis.merge(
                    service_dept,
                    on="رمز الدائرة",
                    how="left"
                )
            )

        # Average salary
        if (
            "مجموع الراتب"
            in filtered_df.columns
        ):

            salary_dept = (
                filtered_df
                .groupby(
                    "رمز الدائرة"
                )[
                    "مجموع الراتب"
                ]
                .mean()
                .round(2)
                .reset_index(
                    name="متوسط مجموع الراتب"
                )
            )

            dept_analysis = (
                dept_analysis.merge(
                    salary_dept,
                    on="رمز الدائرة",
                    how="left"
                )
            )

        # Top reason
        if (
            "سبب انتهاء الخدمة"
            in filtered_df.columns
        ):

            reason_dept = (
                filtered_df
                .groupby(
                    "رمز الدائرة"
                )[
                    "سبب انتهاء الخدمة"
                ]
                .agg(
                    safe_mode
                )
                .reset_index(
                    name="أكثر سبب انتهاء خدمة"
                )
            )

            dept_analysis = (
                dept_analysis.merge(
                    reason_dept,
                    on="رمز الدائرة",
                    how="left"
                )
            )

        total_dept_cases = (
            dept_analysis[
                "عدد المنتهية خدماتهم"
            ].sum()
        )

        dept_analysis[
            "النسبة من الإجمالي %"
        ] = (
            dept_analysis[
                "عدد المنتهية خدماتهم"
            ]
            / total_dept_cases
            * 100
        ).round(2)

        dept_analysis = (
            dept_analysis
            .sort_values(
                "عدد المنتهية خدماتهم",
                ascending=False
            )
        )

        st.dataframe(
            dept_analysis,
            use_container_width=True,
            hide_index=True
        )

        show_bar_chart(
            dept_analysis,
            "رمز الدائرة",
            value="عدد المنتهية خدماتهم",
            title="المنتهية خدماتهم حسب الدائرة",
            horizontal=True,
            key="dept_chart"
        )

        # Organizational units
        if (
            "الوحدة التنظيمية"
            in filtered_df.columns
        ):

            st.markdown(
                "### الوحدات التنظيمية"
            )

            unit_table = (
                build_count_table(
                    filtered_df,
                    "الوحدة التنظيمية"
                )
            )

            st.dataframe(
                unit_table,
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# OPTION 3 - END REASONS
# ============================================================

elif analysis_option == "تحليل أسباب انتهاء الخدمة":

    st.subheader(
        "📋 تحليل أسباب انتهاء الخدمة"
    )

    if (
        "سبب انتهاء الخدمة"
        not in filtered_df.columns
    ):

        st.warning(
            "حقل سبب انتهاء الخدمة غير موجود."
        )

    else:

        reasons_table = (
            build_count_table(
                filtered_df,
                "سبب انتهاء الخدمة"
            )
        )

        total_reasons = (
            reasons_table[
                "العدد"
            ].sum()
        )

        st.metric(
            "إجمالي حالات انتهاء الخدمة",
            f"{total_reasons:,}"
        )

        st.markdown(
            "### الأسباب والعدد والنسبة"
        )

        st.dataframe(
            reasons_table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "سبب انتهاء الخدمة":
                    st.column_config.TextColumn(
                        "سبب انتهاء الخدمة"
                    ),
                "العدد":
                    st.column_config.NumberColumn(
                        "العدد",
                        format="%d"
                    ),
                "النسبة %":
                    st.column_config.NumberColumn(
                        "النسبة %",
                        format="%.2f%%"
                    )
            }
        )

        if not reasons_table.empty:

            top_reason = (
                reasons_table.iloc[0]
            )

            c1, c2 = st.columns(2)

            c1.metric(
                "أكثر سبب لانتهاء الخدمة",
                str(
                    top_reason[
                        "سبب انتهاء الخدمة"
                    ]
                )
            )

            c2.metric(
                "عدد حالات السبب الأعلى",
                f"{int(top_reason['العدد']):,}"
            )

        # Reasons by department
        if (
            "رمز الدائرة"
            in filtered_df.columns
        ):

            st.markdown(
                "### أسباب انتهاء الخدمة حسب الدائرة"
            )

            reason_dept_table = (
                filtered_df
                .assign(
                    سبب_منظف=(
                        filtered_df[
                            "سبب انتهاء الخدمة"
                        ]
                        .fillna(
                            "غير محدد"
                        )
                    )
                )
                .groupby(
                    [
                        "رمز الدائرة",
                        "سبب_منظف"
                    ]
                )
                .size()
                .reset_index(
                    name="العدد"
                )
                .rename(
                    columns={
                        "سبب_منظف":
                            "سبب انتهاء الخدمة"
                    }
                )
            )

            dept_totals = (
                reason_dept_table
                .groupby(
                    "رمز الدائرة"
                )[
                    "العدد"
                ]
                .transform(
                    "sum"
                )
            )

            reason_dept_table[
                "النسبة داخل الدائرة %"
            ] = (
                reason_dept_table[
                    "العدد"
                ]
                / dept_totals
                * 100
            ).round(2)

            reason_dept_table = (
                reason_dept_table
                .sort_values(
                    [
                        "رمز الدائرة",
                        "العدد"
                    ],
                    ascending=[
                        True,
                        False
                    ]
                )
            )

            st.dataframe(
                reason_dept_table,
                use_container_width=True,
                hide_index=True
            )

        # Reasons by year
        if (
            "سنة انتهاء الخدمة"
            in filtered_df.columns
        ):

            st.markdown(
                "### أسباب انتهاء الخدمة حسب السنة"
            )

            reason_year = (
                filtered_df
                .dropna(
                    subset=[
                        "سنة انتهاء الخدمة"
                    ]
                )
                .groupby(
                    [
                        "سنة انتهاء الخدمة",
                        "سبب انتهاء الخدمة"
                    ],
                    dropna=False
                )
                .size()
                .reset_index(
                    name="العدد"
                )
            )

            if not reason_year.empty:

                reason_year[
                    "سنة انتهاء الخدمة"
                ] = (
                    reason_year[
                        "سنة انتهاء الخدمة"
                    ]
                    .astype(int)
                )

                year_totals = (
                    reason_year
                    .groupby(
                        "سنة انتهاء الخدمة"
                    )[
                        "العدد"
                    ]
                    .transform(
                        "sum"
                    )
                )

                reason_year[
                    "النسبة داخل السنة %"
                ] = (
                    reason_year[
                        "العدد"
                    ]
                    / year_totals
                    * 100
                ).round(2)

                st.dataframe(
                    reason_year,
                    use_container_width=True,
                    hide_index=True
                )


# ============================================================
# OPTION 4 - RESIGNATIONS
# ============================================================

elif analysis_option == "تحليل الاستقالات":

    st.subheader(
        "🚪 تحليل الاستقالات"
    )

    if resignation_df.empty:

        st.info(
            "لا توجد حالات استقالة "
            "ضمن الفلاتر الحالية."
        )

    else:

        if (
            "مدة الخدمة بالسنوات"
            in resignation_df.columns
        ):

            avg_resignation_service = (
                resignation_df[
                    "مدة الخدمة بالسنوات"
                ].mean()
            )

        else:

            avg_resignation_service = (
                np.nan
            )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "إجمالي عدد المستقيلين",
            f"{resignation_count:,}"
        )

        c2.metric(
            "نسبة الاستقالات من حالات انتهاء الخدمة",
            f"{resignation_pct:.2f}%"
        )

        if pd.notna(
            avg_resignation_service
        ):

            c3.metric(
                "متوسط مدة الخدمة قبل الاستقالة",
                f"{avg_resignation_service:.2f} سنة"
            )

        else:

            c3.metric(
                "متوسط مدة الخدمة قبل الاستقالة",
                "—"
            )

        # ----------------------------------------------------
        # RESIGNATIONS BY DEPARTMENT
        # ----------------------------------------------------

        if (
            "رمز الدائرة"
            in resignation_df.columns
        ):

            st.markdown(
                "### 🏢 عدد المستقيلين حسب الدائرة"
            )

            if (
                "الرقم الوظيفي"
                in resignation_df.columns
            ):

                resignation_dept = (
                    resignation_df
                    .groupby(
                        "رمز الدائرة"
                    )[
                        "الرقم الوظيفي"
                    ]
                    .nunique()
                    .reset_index(
                        name="عدد المستقيلين"
                    )
                )

            else:

                resignation_dept = (
                    resignation_df
                    .groupby(
                        "رمز الدائرة"
                    )
                    .size()
                    .reset_index(
                        name="عدد المستقيلين"
                    )
                )

            resignation_dept = (
                resignation_dept
                .sort_values(
                    "عدد المستقيلين",
                    ascending=False
                )
                .reset_index(
                    drop=True
                )
            )

            total_resignations = (
                resignation_dept[
                    "عدد المستقيلين"
                ].sum()
            )

            resignation_dept[
                "النسبة من إجمالي الاستقالات %"
            ] = (
                resignation_dept[
                    "عدد المستقيلين"
                ]
                / total_resignations
                * 100
            ).round(2)

            resignation_dept.insert(
                0,
                "الترتيب",
                range(
                    1,
                    len(
                        resignation_dept
                    ) + 1
                )
            )

            st.dataframe(
                resignation_dept,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "الترتيب":
                        st.column_config.NumberColumn(
                            "الترتيب",
                            format="%d"
                        ),
                    "رمز الدائرة":
                        st.column_config.TextColumn(
                            "الدائرة"
                        ),
                    "عدد المستقيلين":
                        st.column_config.NumberColumn(
                            "عدد المستقيلين",
                            format="%d"
                        ),
                    "النسبة من إجمالي الاستقالات %":
                        st.column_config.NumberColumn(
                            "النسبة من إجمالي الاستقالات",
                            format="%.2f%%"
                        )
                }
            )

            show_bar_chart(
                resignation_dept,
                "رمز الدائرة",
                value="عدد المستقيلين",
                title="عدد المستقيلين حسب الدائرة",
                horizontal=True,
                key="resignation_dept_chart"
            )

        # ----------------------------------------------------
        # RESIGNATION REASONS
        # ----------------------------------------------------

        if (
            "سبب الاستقالة"
            in resignation_df.columns
        ):

            st.markdown(
                "### 📋 أسباب الاستقالة"
            )

            resignation_reasons = (
                build_count_table(
                    resignation_df,
                    "سبب الاستقالة"
                )
            )

            st.dataframe(
                resignation_reasons,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "سبب الاستقالة":
                        st.column_config.TextColumn(
                            "سبب الاستقالة"
                        ),
                    "العدد":
                        st.column_config.NumberColumn(
                            "العدد",
                            format="%d"
                        ),
                    "النسبة %":
                        st.column_config.NumberColumn(
                            "النسبة %",
                            format="%.2f%%"
                        )
                }
            )

        # ----------------------------------------------------
        # RESIGNATION REASON BY DEPARTMENT
        # ----------------------------------------------------

        if (
            "رمز الدائرة"
            in resignation_df.columns
            and
            "سبب الاستقالة"
            in resignation_df.columns
        ):

            st.markdown(
                "### 📊 أسباب الاستقالة حسب الدائرة"
            )

            resignation_reason_dept = (
                resignation_df.copy()
            )

            resignation_reason_dept[
                "سبب الاستقالة"
            ] = (
                resignation_reason_dept[
                    "سبب الاستقالة"
                ]
                .fillna(
                    "غير محدد"
                )
                .astype(str)
                .str.strip()
            )

            if (
                "الرقم الوظيفي"
                in resignation_reason_dept.columns
            ):

                resignation_reason_dept = (
                    resignation_reason_dept
                    .groupby(
                        [
                            "رمز الدائرة",
                            "سبب الاستقالة"
                        ]
                    )[
                        "الرقم الوظيفي"
                    ]
                    .nunique()
                    .reset_index(
                        name="العدد"
                    )
                )

            else:

                resignation_reason_dept = (
                    resignation_reason_dept
                    .groupby(
                        [
                            "رمز الدائرة",
                            "سبب الاستقالة"
                        ]
                    )
                    .size()
                    .reset_index(
                        name="العدد"
                    )
                )

            dept_resignation_totals = (
                resignation_reason_dept
                .groupby(
                    "رمز الدائرة"
                )[
                    "العدد"
                ]
                .transform(
                    "sum"
                )
            )

            resignation_reason_dept[
                "النسبة داخل الدائرة %"
            ] = (
                resignation_reason_dept[
                    "العدد"
                ]
                / dept_resignation_totals
                * 100
            ).round(2)

            resignation_reason_dept = (
                resignation_reason_dept
                .sort_values(
                    [
                        "رمز الدائرة",
                        "العدد"
                    ],
                    ascending=[
                        True,
                        False
                    ]
                )
            )

            st.dataframe(
                resignation_reason_dept,
                use_container_width=True,
                hide_index=True
            )

        # ----------------------------------------------------
        # RESIGNATION TREND
        # ----------------------------------------------------

        if (
            "سنة انتهاء الخدمة"
            in resignation_df.columns
        ):

            st.markdown(
                "### 📈 اتجاه الاستقالات عبر السنوات"
            )

            resignation_trend = (
                resignation_df
                .dropna(
                    subset=[
                        "سنة انتهاء الخدمة"
                    ]
                )
                .groupby(
                    "سنة انتهاء الخدمة"
                )
                .size()
                .reset_index(
                    name="عدد الاستقالات"
                )
                .sort_values(
                    "سنة انتهاء الخدمة"
                )
            )

            if not resignation_trend.empty:

                resignation_trend[
                    "سنة انتهاء الخدمة"
                ] = (
                    resignation_trend[
                        "سنة انتهاء الخدمة"
                    ]
                    .astype(int)
                )

                fig = px.line(
                    resignation_trend,
                    x="سنة انتهاء الخدمة",
                    y="عدد الاستقالات",
                    markers=True,
                    text="عدد الاستقالات"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key="resignation_yearly_trend"
                )


# ============================================================
# OPTION 5 - SERVICE LENGTH
# ============================================================

elif analysis_option == "تحليل مدة الخدمة":

    st.subheader(
        "⏳ تحليل مدة الخدمة"
    )

    if (
        "مدة الخدمة بالسنوات"
        not in filtered_df.columns
    ):

        st.warning(
            "لا يمكن حساب مدة الخدمة."
        )

    else:

        service_values = (
            filtered_df[
                "مدة الخدمة بالسنوات"
            ]
            .dropna()
        )

        if service_values.empty:

            st.warning(
                "لا توجد مدد خدمة صالحة."
            )

        else:

            c1, c2, c3, c4 = (
                st.columns(4)
            )

            c1.metric(
                "متوسط مدة الخدمة",
                f"{service_values.mean():.2f} سنة"
            )

            c2.metric(
                "وسيط مدة الخدمة",
                f"{service_values.median():.2f} سنة"
            )

            c3.metric(
                "أقل مدة خدمة",
                f"{service_values.min():.2f} سنة"
            )

            c4.metric(
                "أعلى مدة خدمة",
                f"{service_values.max():.2f} سنة"
            )

            service_table = (
                build_count_table(
                    filtered_df,
                    "فئة مدة الخدمة"
                )
            )

            st.markdown(
                "### توزيع مدة الخدمة"
            )

            st.dataframe(
                service_table,
                use_container_width=True,
                hide_index=True
            )

            show_bar_chart(
                service_table,
                "فئة مدة الخدمة",
                title="توزيع مدة الخدمة",
                horizontal=False,
                key="service_length_chart"
            )

            if (
                "رمز الدائرة"
                in filtered_df.columns
            ):

                st.markdown(
                    "### متوسط مدة الخدمة حسب الدائرة"
                )

                service_dept = (
                    filtered_df
                    .groupby(
                        "رمز الدائرة"
                    )[
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
                    "الدائرة",
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
# OPTION 6 - JOB GROUPS
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

    for index, col in enumerate(
        job_columns
    ):

        if col in filtered_df.columns:

            st.markdown(
                f"### {col}"
            )

            table = (
                build_count_table(
                    filtered_df,
                    col
                )
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
                horizontal=True,
                key=f"job_group_{index}"
            )


# ============================================================
# OPTION 7 - GRADES AND TITLES
# ============================================================

elif analysis_option == "تحليل الدرجات والمسميات الوظيفية":

    st.subheader(
        "💼 تحليل الدرجات والمسميات الوظيفية"
    )

    if (
        "الدرجة الوظيفية"
        in filtered_df.columns
    ):

        st.markdown(
            "### الدرجات الوظيفية"
        )

        grade_table = (
            build_count_table(
                filtered_df,
                "الدرجة الوظيفية"
            )
        )

        st.dataframe(
            grade_table,
            use_container_width=True,
            hide_index=True
        )

        show_bar_chart(
            grade_table,
            "الدرجة الوظيفية",
            title="انتهاء الخدمة حسب الدرجة",
            horizontal=True,
            key="grade_chart"
        )

    if (
        "المسمى الوظيفي"
        in filtered_df.columns
    ):

        st.markdown(
            "### المسميات الوظيفية"
        )

        title_table = (
            build_count_table(
                filtered_df,
                "المسمى الوظيفي"
            )
        )

        st.dataframe(
            title_table,
            use_container_width=True,
            hide_index=True
        )

        show_bar_chart(
            title_table.head(20),
            "المسمى الوظيفي",
            title="أعلى المسميات الوظيفية",
            horizontal=True,
            key="title_chart"
        )


# ============================================================
# OPTION 8 - NATIONALITY
# ============================================================

elif analysis_option == "تحليل الجنسية":

    st.subheader(
        "🌍 تحليل الجنسية"
    )

    if (
        "فئة الجنسية"
        in filtered_df.columns
    ):

        st.markdown(
            "### فئة الجنسية"
        )

        nationality_category = (
            build_count_table(
                filtered_df,
                "فئة الجنسية"
            )
        )

        st.dataframe(
            nationality_category,
            use_container_width=True,
            hide_index=True
        )

    if (
        "الجنسية"
        in filtered_df.columns
    ):

        st.markdown(
            "### الجنسية"
        )

        nationality_table = (
            build_count_table(
                filtered_df,
                "الجنسية"
            )
        )

        st.dataframe(
            nationality_table,
            use_container_width=True,
            hide_index=True
        )

        show_bar_chart(
            nationality_table.head(20),
            "الجنسية",
            title="توزيع الجنسيات",
            horizontal=True,
            key="nationality_chart"
        )


# ============================================================
# OPTION 9 - FINANCIAL
# ============================================================

elif analysis_option == "التحليل المالي":

    st.subheader(
        "💰 التحليل المالي"
    )

    salary_available = (
        "مجموع الراتب"
        in filtered_df.columns
    )

    basic_available = (
        "الراتب الاساسي"
        in filtered_df.columns
    )

    if (
        not salary_available
        and
        not basic_available
    ):

        st.warning(
            "حقول الرواتب غير موجودة."
        )

    else:

        c1, c2, c3, c4 = (
            st.columns(4)
        )

        if salary_available:

            c1.metric(
                "إجمالي مجموع الرواتب",
                f"{filtered_df['مجموع الراتب'].sum():,.2f}"
            )

            c2.metric(
                "متوسط مجموع الراتب",
                f"{filtered_df['مجموع الراتب'].mean():,.2f}"
            )

        else:

            c1.metric(
                "إجمالي مجموع الرواتب",
                "—"
            )

            c2.metric(
                "متوسط مجموع الراتب",
                "—"
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

        else:

            c3.metric(
                "إجمالي الراتب الأساسي",
                "—"
            )

            c4.metric(
                "متوسط الراتب الأساسي",
                "—"
            )

        if (
            "رمز الدائرة"
            in filtered_df.columns
            and
            salary_available
        ):

            st.markdown(
                "### الرواتب حسب الدائرة"
            )

            financial_dept = (
                filtered_df
                .groupby(
                    "رمز الدائرة"
                )
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

            st.dataframe(
                financial_dept,
                use_container_width=True,
                hide_index=True
            )

        if (
            "سبب انتهاء الخدمة"
            in filtered_df.columns
            and
            salary_available
        ):

            st.markdown(
                "### الرواتب حسب سبب انتهاء الخدمة"
            )

            salary_reason = (
                filtered_df
                .groupby(
                    "سبب انتهاء الخدمة",
                    dropna=False
                )[
                    "مجموع الراتب"
                ]
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
# OPTION 10 - TRENDS
# ============================================================

elif analysis_option == "الاتجاهات الزمنية Trends":

    st.subheader(
        "📈 الاتجاهات الزمنية"
    )

    if (
        "سنة انتهاء الخدمة"
        not in filtered_df.columns
    ):

        st.warning(
            "تاريخ انتهاء الخدمة غير متوفر."
        )

    else:

        yearly = (
            filtered_df
            .dropna(
                subset=[
                    "سنة انتهاء الخدمة"
                ]
            )
            .groupby(
                "سنة انتهاء الخدمة"
            )
            .size()
            .reset_index(
                name="العدد"
            )
            .sort_values(
                "سنة انتهاء الخدمة"
            )
        )

        if not yearly.empty:

            yearly[
                "سنة انتهاء الخدمة"
            ] = (
                yearly[
                    "سنة انتهاء الخدمة"
                ]
                .astype(int)
            )

            st.markdown(
                "### الاتجاه السنوي"
            )

            st.dataframe(
                yearly,
                use_container_width=True,
                hide_index=True
            )

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
                key="trend_yearly"
            )

        if (
            "شهر انتهاء الخدمة"
            in filtered_df.columns
        ):

            monthly = (
                filtered_df
                .dropna(
                    subset=[
                        "تاريخ انتهاء الخدمة"
                    ]
                )
                .groupby(
                    "شهر انتهاء الخدمة"
                )
                .size()
                .reset_index(
                    name="العدد"
                )
                .sort_values(
                    "شهر انتهاء الخدمة"
                )
            )

            st.markdown(
                "### الاتجاه الشهري"
            )

            st.dataframe(
                monthly,
                use_container_width=True,
                hide_index=True
            )

            fig = px.line(
                monthly,
                x="شهر انتهاء الخدمة",
                y="العدد",
                markers=True
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key="trend_monthly"
            )

        if (
            "رمز الدائرة"
            in filtered_df.columns
        ):

            st.markdown(
                "### الاتجاه السنوي حسب الدائرة"
            )

            yearly_dept = (
                filtered_df
                .dropna(
                    subset=[
                        "سنة انتهاء الخدمة"
                    ]
                )
                .groupby(
                    [
                        "سنة انتهاء الخدمة",
                        "رمز الدائرة"
                    ]
                )
                .size()
                .reset_index(
                    name="العدد"
                )
            )

            if not yearly_dept.empty:

                yearly_dept[
                    "سنة انتهاء الخدمة"
                ] = (
                    yearly_dept[
                        "سنة انتهاء الخدمة"
                    ]
                    .astype(int)
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
                    key="trend_dept"
                )


# ============================================================
# OPTION 11 - EARLY EXIT
# ============================================================

elif analysis_option == "الخروج المبكر من الخدمة":

    st.subheader(
        "⚠️ تحليل الخروج المبكر من الخدمة"
    )

    if (
        "مدة الخدمة بالسنوات"
        not in filtered_df.columns
    ):

        st.warning(
            "لا يمكن حساب الخروج المبكر."
        )

    else:

        valid_service_df = (
            filtered_df[
                filtered_df[
                    "مدة الخدمة بالسنوات"
                ].notna()
            ]
            .copy()
        )

        if valid_service_df.empty:

            st.warning(
                "لا توجد مدد خدمة صالحة."
            )

        else:

            less_1 = (
                valid_service_df[
                    valid_service_df[
                        "مدة الخدمة بالسنوات"
                    ] < 1
                ]
            )

            less_3 = (
                valid_service_df[
                    valid_service_df[
                        "مدة الخدمة بالسنوات"
                    ] < 3
                ]
            )

            total_valid = (
                count_unique_employees(
                    valid_service_df
                )
            )

            less_1_count = (
                count_unique_employees(
                    less_1
                )
            )

            less_3_count = (
                count_unique_employees(
                    less_3
                )
            )

            if total_valid > 0:

                less_1_pct = (
                    less_1_count
                    / total_valid
                    * 100
                )

                less_3_pct = (
                    less_3_count
                    / total_valid
                    * 100
                )

            else:

                less_1_pct = 0
                less_3_pct = 0

            c1, c2, c3, c4 = (
                st.columns(4)
            )

            c1.metric(
                "خروج خلال أول سنة",
                f"{less_1_count:,}"
            )

            c2.metric(
                "نسبة الخروج خلال أول سنة",
                f"{less_1_pct:.2f}%"
            )

            c3.metric(
                "خروج خلال أول 3 سنوات",
                f"{less_3_count:,}"
            )

            c4.metric(
                "نسبة الخروج خلال أول 3 سنوات",
                f"{less_3_pct:.2f}%"
            )

            if (
                "رمز الدائرة"
                in less_3.columns
            ):

                st.markdown(
                    "### الخروج خلال أول 3 سنوات حسب الدائرة"
                )

                early_dept = (
                    build_count_table(
                        less_3,
                        "رمز الدائرة"
                    )
                )

                st.dataframe(
                    early_dept,
                    use_container_width=True,
                    hide_index=True
                )

            if (
                "سبب انتهاء الخدمة"
                in less_3.columns
            ):

                st.markdown(
                    "### أسباب الخروج خلال أول 3 سنوات"
                )

                early_reason = (
                    build_count_table(
                        less_3,
                        "سبب انتهاء الخدمة"
                    )
                )

                st.dataframe(
                    early_reason,
                    use_container_width=True,
                    hide_index=True
                )

            early_resignation = (
                less_3[
                    find_resignation_mask(
                        less_3
                    )
                ]
            )

            early_resignation_count = (
                count_unique_employees(
                    early_resignation
                )
            )

            st.metric(
                "الاستقالات خلال أول 3 سنوات",
                f"{early_resignation_count:,}"
            )


# ============================================================
# OPTION 12 - DATA QUALITY
# ============================================================

elif analysis_option == "جودة البيانات":

    st.subheader(
        "🧹 جودة البيانات"
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    c1.metric(
        "عدد الصفوف الأصلية",
        f"{original_row_count:,}"
    )

    c2.metric(
        "تواريخ تعيين غير مقروءة",
        f"{invalid_hire_dates:,}"
    )

    c3.metric(
        "تواريخ انتهاء غير مقروءة",
        f"{invalid_end_dates:,}"
    )

    c4.metric(
        "انتهاء قبل تاريخ التعيين",
        f"{negative_service:,}"
    )

    quality_rows = []

    for col in EXPECTED_COLUMNS:

        if col in df.columns:

            missing = (
                df[col]
                .isna()
                .sum()
            )

            if len(df) > 0:

                completeness = (
                    1
                    -
                    missing
                    / len(df)
                ) * 100

            else:

                completeness = 0

            quality_rows.append(
                {
                    "الحقل":
                        col,

                    "عدد السجلات":
                        len(df),

                    "القيم المفقودة":
                        missing,

                    "نسبة الاكتمال %":
                        round(
                            completeness,
                            2
                        )
                }
            )

    quality_table = (
        pd.DataFrame(
            quality_rows
        )
    )

    st.dataframe(
        quality_table,
        use_container_width=True,
        hide_index=True
    )

    preview_cols = [
        col
        for col in [
            "الرقم الوظيفي",
            "اسم الموظف",
            "تاريخ التعيين",
            "تاريخ انتهاء الخدمة",
            "مدة الخدمة بالسنوات",
            "مجموع الراتب",
            "الراتب الاساسي"
        ]
        if col in df.columns
    ]

    st.markdown(
        "### عينة بعد التنظيف"
    )

    st.dataframe(
        df[
            preview_cols
        ].head(50),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# EXCEL EXPORT
# ============================================================

st.divider()

st.subheader(
    "📥 تحميل نتائج التحليل"
)

export_sheets = {
    "البيانات بعد التنظيف":
        filtered_df
}


# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

summary_table = pd.DataFrame(
    {
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
            round(
                resignation_pct,
                2
            ),
            round(
                avg_service,
                2
            )
            if pd.notna(
                avg_service
            )
            else np.nan,
            round(
                median_service,
                2
            )
            if pd.notna(
                median_service
            )
            else np.nan,
            round(
                total_salary,
                2
            )
            if pd.notna(
                total_salary
            )
            else np.nan,
            round(
                avg_salary,
                2
            )
            if pd.notna(
                avg_salary
            )
            else np.nan
        ]
    }
)

export_sheets[
    "المؤشرات العامة"
] = summary_table


# ------------------------------------------------------------
# END REASONS EXPORT
# ------------------------------------------------------------

if (
    "سبب انتهاء الخدمة"
    in filtered_df.columns
):

    export_sheets[
        "أسباب انتهاء الخدمة"
    ] = build_count_table(
        filtered_df,
        "سبب انتهاء الخدمة"
    )


# ------------------------------------------------------------
# DEPARTMENT EXPORT
# ------------------------------------------------------------

if (
    "رمز الدائرة"
    in filtered_df.columns
):

    export_sheets[
        "المنتهية حسب الدائرة"
    ] = build_count_table(
        filtered_df,
        "رمز الدائرة"
    )


# ------------------------------------------------------------
# RESIGNATIONS BY DEPARTMENT EXPORT
# ------------------------------------------------------------

if (
    not resignation_df.empty
    and
    "رمز الدائرة"
    in resignation_df.columns
):

    if (
        "الرقم الوظيفي"
        in resignation_df.columns
    ):

        export_resignation_dept = (
            resignation_df
            .groupby(
                "رمز الدائرة"
            )[
                "الرقم الوظيفي"
            ]
            .nunique()
            .reset_index(
                name="عدد المستقيلين"
            )
        )

    else:

        export_resignation_dept = (
            resignation_df
            .groupby(
                "رمز الدائرة"
            )
            .size()
            .reset_index(
                name="عدد المستقيلين"
            )
        )

    export_resignation_dept = (
        export_resignation_dept
        .sort_values(
            "عدد المستقيلين",
            ascending=False
        )
    )

    total_export_resignations = (
        export_resignation_dept[
            "عدد المستقيلين"
        ].sum()
    )

    if total_export_resignations > 0:

        export_resignation_dept[
            "النسبة من إجمالي الاستقالات %"
        ] = (
            export_resignation_dept[
                "عدد المستقيلين"
            ]
            / total_export_resignations
            * 100
        ).round(2)

    export_sheets[
        "الاستقالات حسب الدائرة"
    ] = export_resignation_dept


# ------------------------------------------------------------
# RESIGNATION REASONS EXPORT
# ------------------------------------------------------------

if (
    not resignation_df.empty
    and
    "سبب الاستقالة"
    in resignation_df.columns
):

    export_sheets[
        "أسباب الاستقالة"
    ] = build_count_table(
        resignation_df,
        "سبب الاستقالة"
    )


# ------------------------------------------------------------
# SERVICE LENGTH EXPORT
# ------------------------------------------------------------

if (
    "فئة مدة الخدمة"
    in filtered_df.columns
):

    export_sheets[
        "مدة الخدمة"
    ] = build_count_table(
        filtered_df,
        "فئة مدة الخدمة"
    )


# ------------------------------------------------------------
# YEARLY TREND EXPORT
# ------------------------------------------------------------

if (
    "سنة انتهاء الخدمة"
    in filtered_df.columns
):

    export_yearly = (
        filtered_df
        .dropna(
            subset=[
                "سنة انتهاء الخدمة"
            ]
        )
        .groupby(
            "سنة انتهاء الخدمة"
        )
        .size()
        .reset_index(
            name="العدد"
        )
    )

    if not export_yearly.empty:

        export_yearly[
            "سنة انتهاء الخدمة"
        ] = (
            export_yearly[
                "سنة انتهاء الخدمة"
            ]
            .astype(int)
        )

    export_sheets[
        "الاتجاه السنوي"
    ] = export_yearly


# ------------------------------------------------------------
# CREATE EXCEL
# ------------------------------------------------------------

try:

    excel_output = (
        create_excel_download(
            export_sheets
        )
    )

    st.download_button(
        label=(
            "⬇️ تحميل ملف Excel "
            "بكل النتائج"
        ),
        data=excel_output,
        file_name=(
            "تحليل_المنتهية_خدماتهم.xlsx"
        ),
        mime=(
            "application/"
            "vnd.openxmlformats-"
            "officedocument."
            "spreadsheetml.sheet"
        )
    )

except Exception as e:

    st.error(
        f"تعذر إنشاء ملف Excel: {e}"
    )


# ============================================================
# CLEAN DATA PREVIEW
# ============================================================

with st.expander(
    "🔍 عرض البيانات التفصيلية بعد التنظيف"
):

    st.write(
        "عدد السجلات بعد الفلاتر: "
        f"{len(filtered_df):,}"
    )

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )
