import streamlit as st
import pandas as pd
import re
import numpy as np
import haversine as hs
from haversine import Unit
from rapidfuzz.distance import Levenshtein
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import io
import tempfile

st.set_page_config(layout="wide", page_title="Competitor Matching App")

st.title("Competitor Matching Workflow")

# --- Utility functions ---
def normalize_special_char(input_str):
    """Remove special characters and spaces."""
    return re.sub(r'[ ,\-.;&+@]', '', input_str.strip().lower())

def normalize_articles(input_str):
    """Remove common articles from an input string."""
    return re.sub(r'\b(a|an|the|or|at|and)\b', '', input_str, flags=re.IGNORECASE)

def normalize_property_name(name):
    """Normalize property name."""
    if not isinstance(name, str):
        return ""
    n = name.strip().lower()
    n = normalize_articles(n)
    n = normalize_special_char(n)
    for term in ['apartments', 'apartment', 'townhomes', 'homes']:
        n = n.replace(term, '')
    return n.strip()

def normalize_address(addr):
    """Normalize address."""
    if not isinstance(addr, str):
        return ""
    n = addr.strip().lower()
    n = normalize_articles(n)
    n = normalize_special_char(n)
    return n.strip()

def extract_state(address):
    """Extract state code from address."""
    if not isinstance(address, str):
        return None
    parts = address.split(',')
    if len(parts) < 2:
        return None
    state_zip = parts[-1].strip()
    tokens = state_zip.split()
    return tokens[0] if tokens else None

def get_address_number(address):
    """Extract first token (assumed street number) from an address."""
    tokens = address.split()
    return tokens[0] if tokens else ''

def get_similarity_score(str1, str2):
    """Compute fuzzy similarity score using Levenshtein distance."""
    str1, str2 = str(str1), str(str2)
    max_len = max(len(str1), len(str2))
    if max_len == 0:
        return 1.0
    distance = Levenshtein.distance(str1, str2)
    return 1.0 - (distance / max_len)

def calculate_geospatial_distance(loc1, loc2):
    """Calculate distance between two geolocations using the haversine formula."""
    return hs.haversine(loc1, loc2, Unit.MILES)

# --- Competitor matching function ---
def find_competitors(msa, subject_property_id, property_data, weighting_list):
    """
    Identify competing properties within a given MSA.
    """
    msa_property_data = property_data[property_data['MSA'] == msa].copy()
    st.write(f'Number of properties in {msa}: {msa_property_data.shape[0]}')
    
    # Convert numeric columns to appropriate data types
    numeric_columns = [
        'YEAR_BUILT', 'UNITS', 'LONGTITUDE', 'LATITUDE', 
        'WS_WALKSCORE', 'WS_BIKESCORE', 'CONSTRUCTION_TYPE_NR',
        'UNIT_0_MIX', 'UNIT_1_MIX', 'UNIT_2_MIX', 'UNIT_3_MIX',
        'SQFT_0', 'SQFT_1', 'SQFT_2', 'SQFT_3'
    ]
    
    for col in numeric_columns:
        if col in msa_property_data.columns:
            msa_property_data[col] = pd.to_numeric(msa_property_data[col], errors='coerce')
    
    msa_property_data = pd.concat([msa_property_data] * len(msa_property_data), ignore_index=True)
    
    msa_subject_property_data = msa_property_data[['PROPERTY_ID', 'NAME', 'SUBMARKET', 'LONGTITUDE', 'LATITUDE']].copy()\
                                        .sort_values(by='NAME').reset_index(drop=True)
    msa_comparsion_property_data = msa_property_data.copy()
    
    result = pd.merge(msa_subject_property_data, msa_comparsion_property_data, left_index=True, right_index=True,
                      how='outer', suffixes=('_S', '_C'))
    
    result['distance_in_miles'] = result.apply(lambda x: calculate_geospatial_distance(
            loc1=(x['LATITUDE_S'], x['LONGTITUDE_S']),
            loc2=(x['LATITUDE_C'], x['LONGTITUDE_C'])
        ), axis=1)
    
    comp_set = result[result['PROPERTY_ID_S'] == subject_property_id].reset_index(drop=True)
    subject_property = comp_set[comp_set['PROPERTY_ID_C'] == subject_property_id]

    
    one_mile = comp_set[comp_set['distance_in_miles'] <= 1]
    st.write(f'Density within a mile: {len(one_mile)}')
    
    comp_set = comp_set[comp_set['distance_in_miles'] <= 15]
    
    for col in ['WS_WALKSCORE', 'WS_BIKESCORE']:
        comp_set[col] = pd.to_numeric(comp_set[col], errors='coerce')
        subject_property[col] = pd.to_numeric(subject_property[col], errors='coerce')
    
    if not subject_property.empty:
        subject_walkscore = subject_property['WS_WALKSCORE'].values[0]
        subject_bikescore = subject_property['WS_BIKESCORE'].values[0]
        comp_set = comp_set[(comp_set['WS_WALKSCORE'] >= subject_walkscore - 20) &
                            (comp_set['WS_WALKSCORE'] <= subject_walkscore + 20)]
        comp_set = comp_set[(comp_set['WS_BIKESCORE'] >= subject_bikescore - 20) &
                            (comp_set['WS_BIKESCORE'] <= subject_bikescore + 20)]
    
    comp_set = comp_set.sort_values('distance_in_miles').reset_index(drop=True)
    subject_index = 0
    
    training_cols = ['distance_in_miles', 'YEAR_BUILT', 'CONSTRUCTION_TYPE_NR', 'WS_BIKESCORE', 'WS_WALKSCORE',
                     'UNITS', 'UNIT_0_MIX', 'UNIT_1_MIX', 'UNIT_2_MIX', 'UNIT_3_MIX', 
                     'SQFT_0', 'SQFT_1', 'SQFT_2', 'SQFT_3']
    training_cols = [col for col in training_cols if col in comp_set.columns]
    training_data = comp_set.select_dtypes(include='number')[training_cols]
    training_data = training_data.apply(lambda x: x.fillna(x.mean()), axis=0)
    
    scaler = StandardScaler()
    training_data_scaled = scaler.fit_transform(training_data)
    weighting_array = np.array(weighting_list)
    training_data_scaled = training_data_scaled * weighting_array
    
    neigh = NearestNeighbors(n_neighbors=7)
    neigh.fit(training_data_scaled)
    neighbors_arr = neigh.kneighbors([training_data_scaled[subject_index]], 7)
    neighbors_lst = list(neighbors_arr[1][0])
    distance_score = list(neighbors_arr[0][0])
    similarity_score = [max(0, 100 - x) for x in distance_score]
    
    competitors = comp_set.iloc[neighbors_lst, :][['PROPERTY_ID_S', 'NAME_S', 'NAME_C', 'SUBMARKET_C',
                                                     'MSA', 'YEAR_BUILT', 'distance_in_miles', 'CONSTRUCTION_TYPE',
                                                     'WS_BIKESCORE', 'WS_WALKSCORE', 'UNITS', 'UNIT_0_MIX',
                                                     'UNIT_1_MIX', 'UNIT_2_MIX', 'UNIT_3_MIX', 'Latest survey_date']]

    competitors = comp_set.iloc[neighbors_lst, :][['PROPERTY_ID_S', 'NAME_S', 'NAME_C', 'SUBMARKET_C',
                                                     'MSA', 'YEAR_BUILT', 'distance_in_miles', 'CONSTRUCTION_TYPE',
                                                     'WS_BIKESCORE', 'WS_WALKSCORE', 'UNITS', 'UNIT_0_MIX',
                                                     'UNIT_1_MIX', 'UNIT_2_MIX', 'UNIT_3_MIX', 'Latest survey_date',  'Sourced', 'Complete', 
                                                     'Integrated', 'Viable', 'DataQuality', 'Is_Older_Than_15_Days', 'PMS_Guess']]
                                                     
    
    competitors['Similarity_Score'] = similarity_score
    competitors['nr_of_props_in_comp_set'] = len(comp_set)
    competitors['nr_of_props_within_a_mile'] = len(one_mile)
    
    output_columns = ['PROPERTY_ID_S', 'NAME_S', 'NAME_C', 'YEAR_BUILT', 'distance_in_miles', 
                      'CONSTRUCTION_TYPE', 'WS_BIKESCORE', 'WS_WALKSCORE', 'UNITS', 
                      'Similarity_Score', 'Latest survey_date', 'Sourced', 'Complete', 'Integrated', 'Viable',
                      'DataQuality', 'Is_Older_Than_15_Days', 'PMS_Guess']
    
    
    return competitors[output_columns]

## PROXIMITY - 35%
weight_distance_in_miles = 35

## YEAR_BUILT - 20%
weight_year_build = 20

#CONSTRUCTION_TYPE - 7.5%
weight_construction_type = 7.5

## BIKE & WALK SCORE - 15%
weight_ws_bikescore = 7.5
weight_ws_walkscore = 7.5

## UNITS 15%
weight_units = 3
weight_unit_mix_0 = 3
weight_unit_mix_1 = 3
weight_unit_mix_2 = 3
weight_unit_mix_3 = 3

## SQFT 7.5%
weight_sqft_0 = 1.875
weight_sqft_1 = 1.875
weight_sqft_2 = 1.875
weight_sqft_3 = 1.875

## NER 0%
weight_ner_0 = 0
weight_ner_1 = 0
weight_ner_2 = 0
weight_ner_3 = 0

weighting_list = [weight_distance_in_miles, weight_year_build, weight_construction_type, weight_ws_bikescore, 
                 weight_ws_walkscore, weight_units, weight_unit_mix_0, weight_unit_mix_1, weight_unit_mix_2, 
                 weight_unit_mix_3, weight_sqft_0, weight_sqft_1, weight_sqft_2, weight_sqft_3]
                 #weight_ner_0, weight_ner_1, weight_ner_2, weight_ner_3]

# --- Load static data ---
@st.cache_data
def load_static_data():
    radix_file = 'data_preperation_for_comp_match.csv'
    # Both property_data and radix_df are from the same static source.
    radix_df = pd.read_csv(radix_file, dtype=str)
    property_data = pd.read_csv(radix_file, dtype=str)
    # Combine address parts and normalize.
    radix_df['combined_address'] = (
        radix_df['ADDRESS'].fillna('') + ', ' +
        radix_df['CITY'].fillna('') + ', ' +
        radix_df['STATE'].fillna('') + ', ' +
        radix_df['ZIP'].fillna('')
    ).str.lower()
    radix_df['addr_norm'] = radix_df['combined_address'].apply(normalize_address)
    radix_df['name_norm'] = radix_df['NAME'].apply(normalize_property_name)
    return radix_df, property_data

radix_df, property_data = load_static_data()

st.sidebar.header("Upload New Properties CSV")
uploaded_new_file = st.sidebar.file_uploader("Upload CSV", type="csv")

if uploaded_new_file is not None:
    try:
        new_df = pd.read_csv(uploaded_new_file, dtype=str)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()
else:
    st.info("Please upload a new properties CSV file to proceed.")
    st.stop()

# Normalize new_df properties.
new_df['address_norm'] = new_df['Address'].apply(normalize_address)
new_df['name_norm'] = new_df['Property Name'].apply(normalize_property_name)
new_df['State'] = new_df['Address'].apply(extract_state)

# --- Mapping: Fuzzy Matching from Radix to New ---
mapping_results = []
for index, radix_row in radix_df.iterrows():
    best_candidate = None
    best_candidate_tuple = (0, 0)
    candidates = new_df[new_df['State'] == radix_row['STATE']]
    for s_index, new_row in candidates.iterrows():
        addr_score = get_similarity_score(radix_row['addr_norm'], new_row['address_norm'])
        name_score = get_similarity_score(radix_row['name_norm'], new_row['name_norm'])
        exact_match = (radix_row['addr_norm'] == new_row['address_norm'])
        cond2 = (addr_score >= 0.75 and name_score >= 0.90)
        same_address_no = (get_address_number(radix_row['addr_norm']) == get_address_number(new_row['address_norm']))
        cond3 = (addr_score >= 0.75 and name_score >= 0.75 and same_address_no)
        cond4 = (addr_score >= 0.99)
        if exact_match or cond2 or cond3 or cond4:
            candidate_tuple = (addr_score, name_score)
            if candidate_tuple > best_candidate_tuple:
                best_candidate_tuple = candidate_tuple
                best_candidate = {
                    'RadixPropertyID': radix_row['PROPERTY_ID'],
                    'RadixName': radix_row['NAME'],
                    'MSA': radix_row['MSA'],
                    'RadixAddress': radix_row['combined_address'],
                    'NewName': new_row['Property Name'],
                    'NewAddress': new_row['Address'],
                    '# Units': new_row['# Units'],
                    'addressScore': addr_score,
                    'nameScore': name_score
                }
    if best_candidate:
        mapping_results.append(best_candidate)
mapping_df = pd.DataFrame(mapping_results)
st.write(f"Mapping complete. Total matched properties: {len(mapping_df)}")
st.dataframe(mapping_df)

# Move competitor analysis into its own function
def perform_competitor_analysis(mapping_df, property_data, weighting_list):
    competitor_results_list = []
    st.header("Competitor Analysis")
    
    for idx, row in mapping_df.iterrows():
        subject_property_id = row['RadixPropertyID']
        msa = row['MSA']
        st.write(f"Processing subject property {subject_property_id} in {msa}...")
        try:
            comp_df = find_competitors(msa, subject_property_id, property_data, weighting_list)
            st.write(comp_df)
        except Exception as e:
            st.error(f"Error processing property {subject_property_id}: {e}")
            continue
        # Append competitor results along with a header row and a separator
        header_row = pd.DataFrame([comp_df.columns.tolist()], columns=comp_df.columns)
        competitor_results_list.append(header_row)
        competitor_results_list.append(comp_df)
        blank_rows = pd.DataFrame([[""] * comp_df.shape[1]] * 2, columns=comp_df.columns)
        competitor_results_list.append(blank_rows)
    
    return competitor_results_list

# Run analysis only if not already in session state
if 'competitor_results' not in st.session_state:
    competitor_results_list = perform_competitor_analysis(mapping_df, property_data, weighting_list)
    if competitor_results_list:
        # Skip the first header since it's redundant
        competitor_results_list = competitor_results_list[1:]
        final_competitors_df = pd.concat(competitor_results_list, ignore_index=True)
        st.session_state.csv_data = final_competitors_df.to_csv(index=False).encode("utf-8")
        st.session_state.competitor_results = True

# Display download button if results exist
if 'csv_data' in st.session_state:
    st.download_button(
        label="Download Competitor Results CSV",
        data=st.session_state.csv_data,
        file_name="competitor_results.csv",
        mime="text/csv"
    )
else:
    st.info("No competitor results generated.")
