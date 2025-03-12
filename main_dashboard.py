import streamlit as st
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit as st
import uuid  # Import uuid to generate unique IDs

def extract_metrics(Small_analysis, line_items, user_input):
    global category_key
    metrics = {}

    categories = {"1": "Stow To Prime", "2": "Each Transfer In","3": "Pick","4": "Chutings","5": "Pack Singles","6": "Pack Multis",}
    selected_category = categories.get(user_input)

    if not selected_category:
        st.error("Invalid input. Please select a value between 1 to 6")
        return None

    category_key = selected_category.lower().replace(" ", "_")
    metrics[category_key] = {}

    for size, name in line_items[selected_category].items():
        size_key = size.lower()
        condition = Small_analysis["LineItem Name"] == name

        try:
            actual_volume = round(float(Small_analysis.loc[condition, "Actual Volume"].iloc[0]), 2)
            actual_hours = round(float(Small_analysis.loc[condition, "Actual Hours"].iloc[0]), 2)
            actual_rate = round(float(Small_analysis.loc[condition, "Actual Rate"].iloc[0]), 2)
            plan_productivity = round(
                float(Small_analysis.loc[condition, "Plan Productivity"].iloc[0]), 2
            )
        except (IndexError, KeyError) as e:
            st.warning(f"Error processing line item '{name}': {e}")
            actual_volume = actual_hours = actual_rate = plan_productivity = None

        learning_curve = [
            round(plan_productivity * factor, 2) if plan_productivity else None
            for factor in [0.7, 0.85, 0.9, 1.0, 1.0]
        ]

        metrics[category_key][size_key] = {
            "Actual Volume": actual_volume,
            "Actual Hours": actual_hours,
            "Actual Rate": actual_rate,
            "Plan Productivity": plan_productivity,
            "Learning Curve": learning_curve,
        }

    return metrics

def process_levels(level_df, metrics, level_index):
    category = category_key
    sizes = ["small", "medium", "large", "total"]

    try:
        for size in sizes:
            level_df[f"{size.capitalize()} UPH Plan"] = metrics[category][size][
                "Learning Curve"
            ][level_index]

        level_df["Meets plan UPH for Small?"] = (
            level_df["Small UPH"] > level_df["Small UPH Plan"]
        )
        level_df["Meets plan UPH for Medium?"] = (
            level_df["Medium UPH"] > level_df["Medium UPH Plan"]
        )
        level_df["Meets plan UPH for Large?"] = (
            level_df["Large UPH"] > level_df["Large UPH Plan"]
        )
        level_df["Meets plan UPH for Total?"] = (
            level_df["Total UPH"] > level_df["Total UPH Plan"]
        )

        level_df["Expected Small units"] = round(
            level_df["Small Hours"] * level_df["Small UPH Plan"], 2
        )
        level_df["Expected Medium units"] = round(
            level_df["Medium Hours"] * level_df["Medium UPH Plan"], 2
        )
        level_df["Expected Large units"] = round(
            level_df["Large Hours"] * level_df["Large UPH Plan"], 2
        )
        level_df["Expected Total units"] = round(
            level_df["Total Hours"] * level_df["Total UPH Plan"], 2
        )

        level_df["Expected Planned Small Hours"] = round(
            level_df["Small Units"] / level_df["Small UPH Plan"], 2
        )
        level_df["Expected Planned Medium Hours"] = round(
            level_df["Medium Units"] / level_df["Medium UPH Plan"], 2
        )
        level_df["Expected Planned Large Hours"] = round(
            level_df["Large Units"] / level_df["Large UPH Plan"], 2
        )
        level_df["Expected Planned Total Hours"] = round(
            level_df["Total Units"] / level_df["Total UPH Plan"], 2
        )

        level_df["Delta Small Hours"] = round(
            level_df["Expected Planned Small Hours"] - level_df["Small Hours"], 2
        )
        level_df["Delta Medium Hours"] = round(
            level_df["Expected Planned Medium Hours"] - level_df["Medium Hours"], 2
        )
        level_df["Delta Large Hours"] = round(
            level_df["Expected Planned Large Hours"] - level_df["Large Hours"], 2
        )
        level_df["Delta Total Hours"] = round(
            level_df["Expected Planned Total Hours"] - level_df["Total Hours"], 2
        )

        level_df["Small % to plan"] = round(
            (level_df["Small Units"] / level_df["Expected Small units"]) * 100, 2
        )
        level_df["Medium % to plan"] = round(
            (level_df["Medium Units"] / level_df["Expected Medium units"]) * 100, 2
        )
        level_df["Large % to plan"] = round(
            (level_df["Large Units"] / level_df["Expected Large units"]) * 100, 2
        )
        level_df["Total % to plan"] = round(
            (level_df["Total Units"] / level_df["Expected Total units"]) * 100, 2
        )

    except KeyError as e:
        st.warning(f"Error processing level data: {e}")

    return level_df

def create_donut_chart(values, labels, hole_size, marker_colors, text, font_size, dimensions,textinfo,hoverinfo):
            fig = go.Figure(go.Pie(
                values=values,
                labels=labels,
                hole=hole_size,  # Donut chart appearance
                marker_colors=marker_colors,  # Color customization
                textinfo=textinfo,  # Remove text
                hoverinfo=hoverinfo # Disable tooltip
            ))
            
            # Layout customization for chart
            fig.update_layout(
                showlegend=False,  # Hide legend
                annotations=[dict(
                    text=text,  # Display text in the center
                    x=0.5, y=0.5,
                    font_size=font_size,
                    showarrow=False
                )],
                width=dimensions[0],
                height=dimensions[1],
                margin=dict(l=0, r=0, t=0, b=0)  # Remove extra margins
            )
            return fig

def display_learning_curve_analysis(category, level, analysis_df, miss, pass_):
    """
    Function to display learning curve analysis for a given category and level.
    
    :param category: str, category name (e.g., 'Small', 'Medium', 'Large', 'Total')
    :param level: int, learning curve level (1, 2, 3, 4, etc.)
    :param analysis_df: DataFrame, filtered analysis data for the level
    :param miss: int, count of employees who missed plan
    :param pass_: int, count of employees who hit plan
    """
    
    st.subheader(f"{category} Learning Curve {level} Analysis")
    
    # Create columns for widgets
    col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9, gap='medium')
    
    # Add headers for each column
    col1.markdown("<h5 style='text-align: center;'>Employee Count</h5>", unsafe_allow_html=True)
    col2.markdown("<h5 style='text-align: center;'>Plan Percentage</h5>", unsafe_allow_html=True)
    
    # Circular progress chart
    progress_chart = create_donut_chart(
        values=[miss, pass_],
        labels=["Missed Plan", "Hit Plan"],
        hole_size=0.8,
        marker_colors=["#1f77b4", "#e6e6e6"],
        text='',
        font_size=20,
        dimensions=(150, 150),
        textinfo='none',
        hoverinfo='label+percent+value'
    )
    
    # Total employees chart
    employee_chart = create_donut_chart(
        values=[len(analysis_df), 0],
        labels=["Employees"],
        hole_size=0.9,
        marker_colors=["#1f77b4"],
        text=f"{len(analysis_df)}",
        font_size=40,
        dimensions=(150, 150),
        textinfo='none',
        hoverinfo='none',
    )
    
    # Generate unique keys using UUID
    unique_id_1 = str(uuid.uuid4())  # Unique key for first plot
    unique_id_2 = str(uuid.uuid4())  # Unique key for second plot
    
    
    
    # Display charts in respective columns with unique keys
    col1.plotly_chart(employee_chart, config={'displayModeBar': False}, key=unique_id_1)
    col2.plotly_chart(progress_chart, config={'displayModeBar': False}, key=unique_id_2)
    
    # Display performance metrics
    col3.metric(value=round(analysis_df[f'Expected Planned {category} Hours'].sum(), 2), label='Total Planned Hours', border=True)
    col4.metric(value=round(analysis_df[f'{category} Hours'].sum(), 2), label='Total Actual Hours', border=True)
    col5.metric(value=round(analysis_df[f'Expected {category} units'].sum(), 2), label='Total Planned Units', border=True)
    col6.metric(value=round(analysis_df[f'{category} Units'].sum(), 2), label='Total Actual Units', border=True)
    col7.metric(value=round(analysis_df[f'{category} UPH Plan'].mean(), 2), label='Avg. Planned UPH', border=True)
    col8.metric(value=round(analysis_df[f'{category} UPH'].mean(), 2), label='Avg. Actual UPH', border=True)
    col9.metric(value=round(analysis_df[f'Delta {category} Hours'].sum(), 2), label='Delta Hours', border=True)

def header_func(logo_url):
    cols1, cols2 = st.columns([1, 9], vertical_alignment= 'center', gap='small')  # This creates a 10% and 90% split
    with cols1:
        st.image(logo_url, width=300)
    with cols2:
        # Align title to the left
        title_html = """
        <div style="display: flex; align-items: center;">
            <h1 style='font-size: 60px;'>
                <span style='color: #6CB4EE;'>Amazon LCY3</span> 
                <span style='color: #5D2491;'>Shift Performance Dashboard</span>
            </h1>
        </div>
        """
        st.markdown(title_html, unsafe_allow_html=True)

def calculate_analysis(data, category):
    # Define column names dynamically based on the category
    expected_units_col = f"Expected {category} units"
    units_col = f"{category} Units"
    expected_hours_col = f"Expected Planned {category} Hours"
    hours_col = f"{category} Hours"
    uph_col = f"{category} UPH"
    uph_plan_col = f"{category} UPH Plan"
    meets_plan_col = f"Meets plan UPH for {category}?"

    # Summing up values
    total_expected_units = round(data[expected_units_col].sum(), 2)
    total_units = round(data[units_col].sum(), 2)
    total_expected_hours = round(data[expected_hours_col].sum(), 2)
    total_hours = round(data[hours_col].sum(), 2)
    mean_uph = round(data[uph_col].mean(), 2)
    mean_plan_uph = round(data[uph_plan_col].mean(), 2)

    # Learning Curve Level Analysis
    levels = {}
    level_dataframes = {}

    for level in range(1, 6):
        level_df = data[data["LC"] == f"Level {level}"]
        level_dataframes[f"Level_{level}"] = level_df  # Store DataFrame for this level
        
        levels[f"Level_{level}_Miss"] = len(level_df[level_df[meets_plan_col] == False])
        levels[f"Level_{level}_Pass"] = len(level_df[level_df[meets_plan_col] == True])

    results = {
        "total_expected_units": total_expected_units,
        "total_units": total_units,
        "total_expected_hours": total_expected_hours,
        "total_hours": total_hours,
        "mean_uph": mean_uph,
        "mean_plan_uph": mean_plan_uph,
        **levels
    }

    return results, level_dataframes  # Return both the aggregated results and the DataFrames


logo_url = "Images/LCY3 Logo.png"
st.set_page_config(page_title='LCY3 Shift Performance Dashboard', page_icon=logo_url, layout="wide")
header_func(logo_url=logo_url)
# st.title("Shift Performance Dashboard")
with st.sidebar:
    process_option = st.radio("Select Process", ("1. Stow To Prime", "2. Each Transfer In","3. Pick", "4. Chutings","5. Pack Singles", "6. Pack Multis"))
    uploaded_file3 = st.file_uploader(
        "Upload Employee Data from PPR", type=['csv']
    )
    uploaded_file1 = st.file_uploader(
        "Upload first CSV or Excel file", type=["csv", "xlsx"]
    )
    uploaded_file2 = st.file_uploader(
        "Upload second CSV or Excel file", type=["csv", "xlsx"]
    )
    

if uploaded_file1 and uploaded_file2:
    header = ['Type', 'ID', 'Name', 'Manager', 'LC', 'Small Hours', 'Medium Hours',
       'Large Hours', 'heavyBulky', 'Total Hours', 'Jobs', 'JPH',
       'Small Units', 'Small UPH', 'Medium Units', 'Medium UPH', 'Large Units',
       'Large UPH', 'Unnamed: 18', 'Unnamed: 19', 'Total Units', 'Total UPH']
    df1 = (
        pd.read_csv(uploaded_file1)
        if uploaded_file1.name.endswith("csv")
        else pd.read_excel(uploaded_file1)
    )
    df2 = (
        pd.read_csv(uploaded_file2,usecols=range(22))
        if uploaded_file2.name.endswith("csv")
        else pd.read_excel(uploaded_file2,usecols=range(22))
    )
    df3 = (
        pd.read_csv(uploaded_file3)
    )

    line_items = {
        "Stow To Prime": {
            "Small": "Each Stow To Prime - Small",
            "Medium": "Each Stow To Prime - Medium",
            "Large": "Each Stow To Prime - Large",
            "Total": "Each Stow to Prime - Total",
        },
        "Each Transfer In": {
            "Small": "Each Transfer In - Small",
            "Medium": "Each Transfer In - Medium",
            "Large": "Each Transfer In - Large",
            "Total": "Each Transfer In - Total",
        },
        "Pick": {
            "Small": "Pick - Small",
            "Medium": "Pick - Medium",
            "Large": "Pick - Large",
            "Total": "Pick - Total",
        },
        "Chutings": {
            "Small": "Chutings - Small",
            "Medium": "Chutings - Medium",
            "Large": "Chutings - Large",
            "Total": "Pack Chuting - Total",
        },
        "Pack Singles": {
            "Small": "Pack Singles - Small",
            "Medium": "Pack Singles - Medium",
            "Large": "Pack Singles - Large",
            "Total": "Pack Singles - Total",
        },
        "Pack Multis": {
            "Small": "Pack Multis - Small",
            "Medium": "Pack Multis - Medium",
            "Large": "Pack Multis - Large",
            "Total": "Pack Multis - Total",
        },
    }

    result = extract_metrics(df1, line_items, process_option[0])

    if result:
        df2.columns = header
        datasheet = df2
        datasheet["LC"] = datasheet["LC"].replace("-", "Level 5")
        datasheet = datasheet.fillna(0)
        # Drop rows where 'Small Hours' equals 0
        datasheet = datasheet[
        (datasheet['Small Hours'] != 0) |
        (datasheet['Medium Hours'] != 0) |
        (datasheet['Large Hours'] != 0) 
        ]

        # Aggregation logic
        aggregation = {
        "Type": "first",
        "ID": "first",
        "Name": "first",
        "Manager": "first",
        "LC": "first",
        "Small Hours": "sum",
        'Medium Hours' : "sum",
        'Large Hours' : "sum",
        'Total Hours':"sum",
        "Small Units": "sum",
        'Medium Units': "sum",
        'Large Units': "sum",
        'Total Units' : "sum",
        'Jobs': "sum",
        "JPH":"mean",
        "Small UPH": "mean",
        'Medium UPH' : "mean",
        'Large UPH' : "mean",
        'Total UPH' : "mean",
        }
        # Group by 'ID' and aggregate
        datasheet= datasheet.groupby("ID", as_index=False).agg(aggregation)
        # Reset the index if needed
        datasheet.reset_index(drop=True, inplace=True)
        levels = {
            f"level_{i}": datasheet.loc[datasheet["LC"] == f"Level {i}"].reset_index()
            for i in range(1, 6)
        }

        for i in range(5):
            levels[f"level_{i+1}"] = process_levels(levels[f"level_{i+1}"], result, i)

        # Concatenate all levels into one DataFrame
        New_data = pd.concat(
            [
                levels["level_1"],
                levels["level_2"],
                levels["level_3"],
                levels["level_4"],
                levels["level_5"],
            ],
            ignore_index=True,
        )
        New_data.fillna(0)
        data = df3
        data=data.iloc[:,0:2]
        data = data.rename(columns = {'Employee ID':'ID'})
        New_data= pd.merge(data,New_data,on='ID')
        # Ensure the ID column is treated as strings
        New_data["ID"] = New_data["ID"].astype(str).str.replace(",", "")
        New_data = New_data.rename(columns = {'User ID':'Username'})

        # Define columns for analysis
        Small_Columns = [
            "Type",
            "Username",
            "ID",
            "Name",
            "Manager",
            "LC",
            "Meets plan UPH for Small?",
            "Small Hours",
            "Expected Planned Small Hours",
            "Small Units",
            "Expected Small units",
            "Small UPH",
            "Small UPH Plan",
            "Delta Small Hours",
            "Small % to plan",
        ]
        Medium_Columns = [
            "Type",
            "Username",
            "ID",
            "Name",
            "Manager",
            "LC",
            "Meets plan UPH for Medium?",
            "Medium Hours",
            "Expected Planned Medium Hours",
            "Medium Units",
            "Expected Medium units",
            "Medium UPH",
            "Medium UPH Plan",
            "Delta Medium Hours",
            "Medium % to plan",
            
        ]
        Large_Columns = [
            "Type",
            "Username",
            "ID",
            "Name",
            "Manager",
            "LC",
            "Meets plan UPH for Large?",
            "Large Hours",
            "Expected Planned Large Hours",
            "Large Units",
            "Expected Large units",
            "Large UPH",
            "Large UPH Plan",
            "Delta Large Hours",
            "Large % to plan",
        ]
        Total_Columns = [
            "Type",
            "Username",
            "ID",
            "Name",
            "Manager",
            "LC",
            "Meets plan UPH for Total?",
            "Total Hours",
            "Expected Planned Total Hours",
            "Total Units",
            "Expected Total units",
            "Total UPH",
            "Total UPH Plan",
            "Delta Total Hours",
            "Total % to plan",
        ]
        combined_columns = [
            "Type",
            "Username",
            "ID",
            "Name",
            "Manager",
            "LC",
            "Meets plan UPH for Small?",
            "Small Hours",
            "Expected Planned Small Hours",
            "Small Units",
            "Expected Small units",
            "Small UPH",
            "Small UPH Plan",
            "Delta Small Hours",
            "Small % to plan",
            
            "Meets plan UPH for Medium?",
            "Medium Hours",
            "Expected Planned Medium Hours",
            "Medium Units",
            "Expected Medium units",
            "Medium UPH",
            "Medium UPH Plan",
            "Delta Medium Hours",
            "Medium % to plan",
            
            "Meets plan UPH for Large?",
            "Large Hours",
            "Expected Planned Large Hours",
            "Large Units",
            "Expected Large units",
            "Large UPH",
            "Large UPH Plan",
            "Delta Large Hours",
            "Large % to plan",
            
            "Meets plan UPH for Total?",
            "Total Hours",
            "Expected Planned Total Hours",
            "Total Units",
            "Expected Total units",
            "Total UPH",
            "Total UPH Plan",
            "Delta Total Hours",
            "Total % to plan",
            
        ]


        # Perform analysis
        Small_analysis = New_data[Small_Columns]
        # Drop rows where 'Small Hours' equals 0
        Small_analysis = Small_analysis[Small_analysis['Small Hours'] != 0]
        # Reset the index if needed
        Small_analysis.reset_index(drop=True, inplace=True)
        
        Medium_analysis = New_data[Medium_Columns]
        Large_analysis = New_data[Large_Columns]
        Total_analysis = New_data[Total_Columns]
        General_analysis = New_data[combined_columns]

        # Sum up values for Small analysis
        total_Small_analysis_expected_units = round(
            Small_analysis["Expected Small units"].sum(), 2
        )
        total_Small_analysis_Small_units = round(Small_analysis["Small Units"].sum(), 2)
        total_Small_analysis_expected_hours = round(
            Small_analysis["Expected Planned Small Hours"].sum(), 2
        )
        total_Small_analysis_Small_hours = round(Small_analysis["Small Hours"].sum(), 2)
        small_uph = round(Small_analysis['Small UPH'].mean(),2)
        plan_small_uph = round(Small_analysis['Small UPH Plan'].mean(),2)

        # Example usage:
        small_analysis_results, small_analysis_levels = calculate_analysis(Small_analysis, "Small")
        medium_analysis_results, medium_analysis_levels = calculate_analysis(Medium_analysis, "Medium")
        large_analysis_results, large_analysis_levels = calculate_analysis(Large_analysis, "Large")
        total_analysis_results, total_analysis_levels = calculate_analysis(Total_analysis, "Total")
        # print(small_analysis_results)
        # Sum up values for Medium analysis
        total_Medium_analysis_expected_units = round(
            Medium_analysis["Expected Medium units"].sum(), 2
        )
        total_Medium_analysis_Medium_units = round(
            Medium_analysis["Medium Units"].sum(), 2
        )
        total_Medium_analysis_expected_hours = round(
            Medium_analysis["Expected Planned Medium Hours"].sum(), 2
        )
        total_Medium_analysis_Medium_hours = round(
            Medium_analysis["Medium Hours"].sum(), 2
        )
        medium_uph = round(Medium_analysis['Medium UPH'].mean(),2)
        plan_medium_uph = round(Medium_analysis['Medium UPH Plan'].mean(),2)

        # Sum up values for Large analysis
        total_Large_analysis_expected_units = round(
            Large_analysis["Expected Large units"].sum(), 2
        )
        total_Large_analysis_Large_units = round(Large_analysis["Large Units"].sum(), 2)
        total_Large_analysis_expected_hours = round(
            Large_analysis["Expected Planned Large Hours"].sum(), 2
        )
        total_Large_analysis_Large_hours = round(Large_analysis["Large Hours"].sum(), 2)
        large_uph = round(Large_analysis['Large UPH'].mean(),2)
        plan_large_uph = round(Large_analysis['Large UPH Plan'].mean(),2)
        
        
        # Sum up values for Total analysis
        total_Total_analysis_expected_units = round(
            Total_analysis["Expected Total units"].sum(), 2
        )
        total_Total_analysis_Total_units = round(Total_analysis["Total Units"].sum(), 2)
        total_Total_analysis_expected_hours = round(
            Total_analysis["Expected Planned Total Hours"].sum(), 2
        )
        total_Total_analysis_Total_hours = round(Total_analysis["Total Hours"].sum(), 2)
        total_uph = round(Total_analysis['Total UPH'].mean(),2)
        plan_total_uph = round(Total_analysis['Total UPH Plan'].mean(),2)


    # Create tabs
    tabs = ["Small", "Medium", "Large", "Total", "General"]
    tab1, tab2, tab3, tab4, tab5 = st.tabs(tabs)
    # Inject CSS to evenly spread tabs
    st.markdown(
        """
        <style>
        .stTabs [data-baseweb="tab"] {
            flex: 1;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    with tab1:
        delta_units = total_Small_analysis_Small_units - total_Small_analysis_expected_units
        delta_hours =  total_Small_analysis_expected_hours - total_Small_analysis_Small_hours
        delta_uph = small_uph - plan_small_uph
        Small_Percent_to_plan = round((total_Small_analysis_Small_units / total_Small_analysis_expected_units) * 100,2)
        small_percent_to_miss = Small_Percent_to_plan - 100
        # Layout configuration
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7, gap='small',border=True,vertical_alignment='center')
        # Display the results
        st.title("Small Analysis")
        # Card 1
        col1.metric(label="Planned Units", value=f"{total_Small_analysis_expected_units:,} units")
        # Card 2
        col2.metric(label="Actual Units", value=f"{total_Small_analysis_Small_units:,} units",delta=f"{round(delta_units,2)} units", delta_color="normal")
        # Card 3
        col3.metric(label="Planned Hours", value=f"{total_Small_analysis_expected_hours} hours")
        # Card 4
        col4.metric(label="Actual Hours", value=f"{total_Small_analysis_Small_hours} hours",delta=f"{round(delta_hours,2)} hours", delta_color="normal")
        # Card 5
        col5.metric(label="Planned UPH", value=f"{plan_small_uph} UPH")
        # Card 5
        col6.metric(label="Actual UPH", value=f"{small_uph} UPH",delta=f"{round(delta_uph,2)} UPH", delta_color="normal")
        # Card 5
        col7.metric(label="% to plan", value=f"{Small_Percent_to_plan} %" ,delta=f"{round(small_percent_to_miss,2)} %", delta_color="normal")
        # Create two columns for multiselect widgets
        col1, col2, col3 = st.columns([7,2,1])
        with col1:
            selected_managers = st.multiselect('Select Managers', Small_analysis['Manager'].unique(),default=Small_analysis['Manager'].unique())

        with col2:
            selected_levels = st.multiselect('Select Level', Small_analysis['LC'].unique(),default=Small_analysis['LC'].unique())
        
        with col3:
            selected_target = st.multiselect('Select Target', Small_analysis['Meets plan UPH for Small?'].unique(),default=Small_analysis['Meets plan UPH for Small?'].unique())

        # Filter DataFrame based on multiple selections
        filtered_df_multiselect = Small_analysis[
            (Small_analysis['Manager'].isin(selected_managers) if selected_managers else True) & 
            (Small_analysis['LC'].isin(selected_levels) if selected_levels else True) &
            (Small_analysis['Meets plan UPH for Small?'].isin(selected_target) if selected_target else True)
        ]
        # Display DataFrame with multiselect functionality
        st.dataframe(filtered_df_multiselect, use_container_width=True, hide_index=True)
        # Example usage:
        display_learning_curve_analysis('Small', 1,small_analysis_levels["Level_1"], small_analysis_results["Level_1_Miss"], small_analysis_results["Level_1_Pass"])
        display_learning_curve_analysis('Small', 2, small_analysis_levels["Level_2"], small_analysis_results["Level_2_Miss"], small_analysis_results["Level_2_Pass"])
        display_learning_curve_analysis('Small', 3, small_analysis_levels["Level_3"], small_analysis_results["Level_3_Miss"], small_analysis_results["Level_3_Pass"])
        display_learning_curve_analysis('Small', 4, small_analysis_levels["Level_4"], small_analysis_results["Level_4_Miss"], small_analysis_results["Level_4_Pass"])
        display_learning_curve_analysis('Small', 5, small_analysis_levels["Level_5"], small_analysis_results["Level_5_Miss"], small_analysis_results["Level_5_Pass"])
        
    with tab2:
        delta_units = total_Medium_analysis_Medium_units - total_Medium_analysis_expected_units
        delta_hours =  total_Medium_analysis_expected_hours - total_Medium_analysis_Medium_hours
        delta_uph = medium_uph - plan_medium_uph
        Medium_Percent_to_plan = round((total_Medium_analysis_Medium_units / total_Medium_analysis_expected_units) * 100,2)
        Medium_percent_to_miss = Medium_Percent_to_plan - 100
        # Layout configuration
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7, gap='large',border=True,vertical_alignment='center')
        st.header("Medium Analysis")
        # Card 1
        col1.metric(label="Planned Units", value=f"{total_Medium_analysis_expected_units:,} units")
        # Card 2
        col2.metric(label=" Actual Units", value=f"{total_Medium_analysis_Medium_units:,} units",delta=f"{round(delta_units,2)} units", delta_color="normal")
        # Card 3
        col3.metric(label="Planned Hours", value=f"{total_Medium_analysis_expected_hours} hours")
        # Card 4
        col4.metric(label="Actual Hours", value=f"{total_Medium_analysis_Medium_hours} hours",delta=f"{round(delta_hours,2)} hours", delta_color="normal")
        # Card 5
        col5.metric(label="Planned UPH", value=f"{plan_medium_uph} UPH")
        # Card 6
        col6.metric(label="Actual UPH", value=f"{medium_uph} UPH",delta=f"{round(delta_uph,2)} UPH", delta_color="normal")
        # Card 7
        col7.metric(label="% to plan", value=f"{Medium_Percent_to_plan} %" ,delta=f"{round(Medium_percent_to_miss,2)} %", delta_color="normal")
        # Create two columns for multiselect widgets
        col1, col2, col3 = st.columns([7,2,1])

        with col1:
            selected_managers = st.multiselect('Select Managers', Medium_analysis['Manager'].unique(),default=Medium_analysis['Manager'].unique(),key=20)

        with col2:
            selected_levels = st.multiselect('Select Level', Medium_analysis['LC'].unique(),default=Medium_analysis['LC'].unique(), key = 21)
        
        with col3:
            selected_target = st.multiselect('Select Target', Medium_analysis['Meets plan UPH for Medium?'].unique(),default=Medium_analysis['Meets plan UPH for Medium?'].unique(),key = 23)

        # Filter DataFrame based on multiple selections
        filtered_df_multiselect = Medium_analysis[
            (Medium_analysis['Manager'].isin(selected_managers) if selected_managers else True) & 
            (Medium_analysis['LC'].isin(selected_levels) if selected_levels else True) &
            (Medium_analysis['Meets plan UPH for Medium?'].isin(selected_target) if selected_target else True)
        ]

        # Display the filtered DataFrame
        st.dataframe(filtered_df_multiselect,  use_container_width=True, hide_index=True)
        
        # Example usage:
        display_learning_curve_analysis('Medium', 1,medium_analysis_levels["Level_1"], medium_analysis_results["Level_1_Miss"], medium_analysis_results["Level_1_Pass"])
        display_learning_curve_analysis('Medium', 2, medium_analysis_levels["Level_2"], medium_analysis_results["Level_2_Miss"], medium_analysis_results["Level_2_Pass"])
        display_learning_curve_analysis('Medium', 3, medium_analysis_levels["Level_3"], medium_analysis_results["Level_3_Miss"], medium_analysis_results["Level_3_Pass"])
        display_learning_curve_analysis('Medium', 4, medium_analysis_levels["Level_4"], medium_analysis_results["Level_4_Miss"], medium_analysis_results["Level_4_Pass"])
        display_learning_curve_analysis('Medium', 5, medium_analysis_levels["Level_5"], medium_analysis_results["Level_5_Miss"], medium_analysis_results["Level_5_Pass"])
        
        
    with tab3:
        delta_units = total_Large_analysis_Large_units - total_Large_analysis_expected_units
        delta_hours =  total_Large_analysis_expected_hours - total_Large_analysis_Large_hours
        delta_uph = large_uph - plan_large_uph
        Large_Percent_to_plan = round((total_Large_analysis_Large_units / total_Large_analysis_expected_units) * 100,2)
        Large_percent_to_miss = Large_Percent_to_plan - 100
        # Layout configuration
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7, gap='large',border=True,vertical_alignment='center')
        st.header("Large Analysis")
        # Card 1
        col1.metric(label="Planned Units", value=f"{total_Large_analysis_expected_units:,} units")
        # Card 2
        col2.metric(label="Actual Units", value=f"{total_Large_analysis_Large_units:,} units",delta=f"{round(delta_units,2)} units", delta_color="normal")
        # Card 3
        col3.metric(label="Planned Hours", value=f"{total_Large_analysis_expected_hours} hours")
        # Card 4
        col4.metric(label="Actual Hours", value=f"{total_Large_analysis_Large_hours} hours",delta=f"{round(delta_hours,2)} hours", delta_color="normal")
        # Card 5
        col5.metric(label="Planned UPH", value=f"{plan_large_uph} UPH")
        # Card 6
        col6.metric(label="Actual UPH", value=f"{large_uph} UPH",delta=f"{round(delta_uph,2)} UPH", delta_color="normal")
        # Card 7
        col7.metric(label="% to plan", value=f"{Large_Percent_to_plan} %" ,delta=f"{round(Large_percent_to_miss,2)} %", delta_color="normal")
        # Create two columns for multiselect widgets
        col1, col2, col3 = st.columns([7,2,1])

        with col1:
            selected_managers = st.multiselect('Select Managers', Large_analysis['Manager'].unique(),default=Large_analysis['Manager'].unique(),key=30)

        with col2:
            selected_levels = st.multiselect('Select Level', Large_analysis['LC'].unique(),default=Large_analysis['LC'].unique(), key = 31)
        
        with col3:
            selected_target = st.multiselect('Select Target', Large_analysis['Meets plan UPH for Large?'].unique(),default=Large_analysis['Meets plan UPH for Large?'].unique(),key = 32)

        # Filter DataFrame based on multiple selections
        filtered_df_multiselect =Large_analysis[
            (Large_analysis['Manager'].isin(selected_managers) if selected_managers else True) & 
            (Large_analysis['LC'].isin(selected_levels) if selected_levels else True) &
            (Large_analysis['Meets plan UPH for Large?'].isin(selected_target) if selected_target else True)
        ]

        # Display the filtered DataFrame
        st.dataframe(filtered_df_multiselect,  use_container_width=True, hide_index=True)
        # Example usage:
        display_learning_curve_analysis('Large', 1,large_analysis_levels["Level_1"], large_analysis_results["Level_1_Miss"], large_analysis_results["Level_1_Pass"])
        display_learning_curve_analysis('Large', 2, large_analysis_levels["Level_2"], large_analysis_results["Level_2_Miss"], large_analysis_results["Level_2_Pass"])
        display_learning_curve_analysis('Large', 3, large_analysis_levels["Level_3"], large_analysis_results["Level_3_Miss"], large_analysis_results["Level_3_Pass"])
        display_learning_curve_analysis('Large', 4, large_analysis_levels["Level_4"], large_analysis_results["Level_4_Miss"], large_analysis_results["Level_4_Pass"])
        display_learning_curve_analysis('Large', 5, large_analysis_levels["Level_5"], large_analysis_results["Level_5_Miss"], large_analysis_results["Level_5_Pass"])


    with tab4:
        delta_units = total_Total_analysis_Total_units - total_Total_analysis_expected_units
        delta_hours =  total_Total_analysis_expected_hours - total_Total_analysis_Total_hours
        delta_uph = total_uph - plan_total_uph
        Total_Percent_to_plan = round((total_Total_analysis_Total_units / total_Total_analysis_expected_units) * 100,2)
        Total_percent_to_miss = Total_Percent_to_plan - 100
        # Layout configuration
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7, gap='large',border=True,vertical_alignment='center')
        st.header("Total Analysis")
        # Card 1
        col1.metric(label="Planned Units", value=f"{total_Total_analysis_expected_units:,} units")
        # Card 2
        col2.metric(label="Actual Units", value=f"{total_Total_analysis_Total_units:,} units",delta=f"{round(delta_units,2)} units", delta_color="normal")
        # Card 3
        col3.metric(label="Planned Hours", value=f"{total_Total_analysis_expected_hours} hours")
        # Card 4
        col4.metric(label="Actual Hours", value=f"{total_Total_analysis_Total_hours} hours",delta=f"{round(delta_hours,2)} hours", delta_color="normal")
        # Card 5
        col5.metric(label="Planned UPH", value=f"{plan_total_uph} UPH")
        # Card 6
        col6.metric(label="Actual UPH", value=f"{total_uph} UPH",delta=f"{round(delta_uph,2)} UPH", delta_color="normal")
        # Card 7
        col7.metric(label="% to plan", value=f"{Total_Percent_to_plan} %" ,delta=f"{round(Total_percent_to_miss,2)} %", delta_color="normal")
        
        # Create two columns for multiselect widgets
        col1, col2, col3 = st.columns([7,2,1])

        with col1:
            selected_managers = st.multiselect('Select Managers', Total_analysis['Manager'].unique(),default=Large_analysis['Manager'].unique(),key=40)

        with col2:
            selected_levels = st.multiselect('Select Level', Total_analysis['LC'].unique(),default=Large_analysis['LC'].unique(), key = 41)
        
        with col3:
            selected_target = st.multiselect('Select Target', Total_analysis['Meets plan UPH for Total?'].unique(),default=Total_analysis['Meets plan UPH for Total?'].unique(),key = 42)

        # Filter DataFrame based on multiple selections
        filtered_df_multiselect = Total_analysis[
            (Total_analysis['Manager'].isin(selected_managers) if selected_managers else True) & 
            (Total_analysis['LC'].isin(selected_levels) if selected_levels else True) &
            (Total_analysis['Meets plan UPH for Total?'].isin(selected_target) if selected_target else True)
        ]

        # Display the filtered DataFrame
        st.dataframe(filtered_df_multiselect,  use_container_width=True, hide_index=True)
        
        display_learning_curve_analysis('Total', 1,total_analysis_levels["Level_1"], total_analysis_results["Level_1_Miss"], total_analysis_results["Level_1_Pass"])
        display_learning_curve_analysis('Total', 2, total_analysis_levels["Level_2"], total_analysis_results["Level_2_Miss"], total_analysis_results["Level_2_Pass"])
        display_learning_curve_analysis('Total', 3, total_analysis_levels["Level_3"], total_analysis_results["Level_3_Miss"], total_analysis_results["Level_3_Pass"])
        display_learning_curve_analysis('Total', 4, total_analysis_levels["Level_4"], total_analysis_results["Level_4_Miss"], total_analysis_results["Level_4_Pass"])
        display_learning_curve_analysis('Total', 5, total_analysis_levels["Level_5"], total_analysis_results["Level_5_Miss"], total_analysis_results["Level_5_Pass"])

    with tab5:
        st.header("General Analysis")
        st.dataframe(General_analysis, use_container_width=True, hide_index=True)
