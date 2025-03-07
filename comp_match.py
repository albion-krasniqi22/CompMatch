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
import requests
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import openai

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

# --- External Data Functions ---
def geocode_address_nominatim(address):
    geolocator = Nominatim(user_agent="competitor_matching_app")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    try:
        location = geocode(address)
        if location:
            return (location.latitude, location.longitude)
    except Exception as e:
        st.error(f"Geocoding error: {e}")
    return (None, None)

def get_walk_and_bike_scores(address, lat, lon, wsapikey="1d6bbbde1a140f0555d22a27ac738828"):
    """
    Query the Walk Score API and return the walk and bike scores.
    """
    base_url = "http://api.walkscore.com/score"
    params = {
        "format": "json",
        "address": address,
        "lat": lat,
        "lon": lon,
        "transit": "1",  
        "bike": "1",     
        "wsapikey": wsapikey
    }
    
    try:
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            walkscore = data.get("walkscore")
            bikescore = data.get("bike", {}).get("score")
            return walkscore, bikescore, data
        else:
            # Silently handle API errors without displaying them
            return None, None, None
    except Exception:
        # Silently handle exceptions without displaying them
        return None, None, None

def get_msa_from_address(address):
    """
    Uses OpenAI's API to extract the Metropolitan Statistical Area (MSA)
    from an address string.
    """
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    client = openai.OpenAI(api_key=openai_api_key)
    msa_list = ['Dallas, TX', 'Philadelphia, PA', 'Tampa, FL', 'San Diego, CA',
       'Austin, TX', 'Colorado Springs, CO', 'Mount Vernon, WA',
       'Palm Bay, FL', 'Houston, TX', 'Spartanburg, SC',
       'Jacksonville, FL', 'San Antonio, TX', 'Charlotte, NC',
       'Atlanta, GA', 'Phoenix, AZ', 'Seattle, WA', 'North Port, FL',
       'Deltona, FL', 'Miami, FL', 'Salt Lake City, UT', 'Wilmington, NC',
       'Portland, OR', 'San Francisco, CA', 'Sacramento, CA',
       'Huntsville, AL', 'Cape Coral, FL', 'Washington, DC',
       'Nashville, TN', 'Los Angeles, CA', 'Detroit, MI', 'Boston, MA',
       'Richmond, VA', 'Denver, CO', 'Urban Honolulu, HI', 'Spokane, WA',
       'Tucson, AZ', 'Myrtle Beach, SC', 'Virginia Beach, VA',
       'Greenville, SC', 'Orlando, FL', 'Kansas City, MO', 'Kenosha, WI',
       'Raleigh, NC', 'Charleston, SC', 'Salinas, CA', 'Chicago, IL',
       'New York, NY', 'Indianapolis, IN', 'Knoxville, TN',
       'Kingsport, TN', 'Cleveland, TN', 'Baltimore, MD',
       'Albuquerque, NM', 'Beaumont, TX', 'Pensacola, FL',
       'Minneapolis, MN', 'Fort Collins, CO', 'Santa Fe, NM',
       'San Jose, CA', 'Longview, TX', 'Port St. Lucie, FL', 'Tyler, TX',
       'Rapid City, SD', 'Greeley, CO', 'Akron, OH', 'Omaha, NE',
       'Lansing, MI', 'Pittsburgh, PA', 'Hot Springs, AR', 'Lubbock, TX',
       'Provo, UT', 'St. George, UT', 'Vallejo, CA', 'Louisville, KY',
       'Lexington, KY', 'Las Vegas, NV', 'Lakeland, FL', 'Boulder, CO',
       'Fayetteville, NC', 'Brunswick, GA', 'Savannah, GA', 'Oxnard, CA',
       'Reno, NV', 'Boise City, ID', 'Flagstaff, AZ', 'Madison, WI',
       'Modesto, CA', 'Twin Falls, ID', 'Wenatchee, WA', 'Jackson, MS',
       'Columbus, OH', 'Flint, MI', 'Ann Arbor, MI', 'Clarksville, TN',
       'Napa, CA', 'Chattanooga, TN', 'Mobile, AL', 'Chico, CA',
       'Little Rock, AR', 'Lake Charles, LA', 'Eugene, OR',
       'Stockton, CA', 'Punta Gorda, FL', 'Gulfport, MS',
       'Hattiesburg, MS', 'St. Louis, MO', 'Fayetteville, AR',
       'Cleveland, OH', 'Greensboro, NC', 'Memphis, TN', 'Salem, OR',
       'Jonesboro, AR', 'Tallahassee, FL', 'Riverside, CA',
       'Hilton Head Island, SC', 'Baton Rouge, LA', 'Augusta, GA',
       'Santa Rosa, CA', 'Olympia, WA', 'New Orleans, LA',
       'Shreveport, LA', 'Hickory, NC', 'Jacksonville, NC',
       'Crestview, FL', 'Ocala, FL', 'Burlington, NC', 'New Bern, NC',
       'Winston, NC', 'Worcester, MA', 'Wildwood, FL', 'Durham, NC',
       'Corpus Christi, TX', 'Birmingham, AL', 'Milwaukee, WI',
       'Asheville, NC', 'Cincinnati, OH', 'Ogden, UT', 'Naples, FL',
       'College Station, TX', 'Bridgeport, CT', 'Trenton, NJ',
       'Athens, GA', 'Santa Cruz, CA', 'Rochester, MN', 'Bremerton, WA',
       'Daphne, AL', 'Panama City, FL', 'El Paso, TX', 'Las Cruces, NM',
       'Jackson, TN', 'Prescott Valley, AZ', 'Dothan, AL',
       'Kennewick, WA', 'Montgomery, AL', 'Oklahoma City, OK',
       'Covington, LA', 'Winchester, VA', 'Midland, TX', 'Medford, OR',
       'Roanoke, VA', 'Warner Robins, GA', 'San Angelo, TX', 'Kokomo, IN',
       'Gainesville, GA', 'Dayton, OH', 'San Luis Obispo, CA',
       'Killeen, TX', 'Tulsa, OK', 'Columbia, SC', 'Johnson City, TN',
       'South Bend, IN', 'Logan, UT', 'Carson City, NV',
       'Charlottesville, VA', 'Dalton, GA', 'Waco, TX', 'Columbus, GA',
       'Columbus, IN', 'Sherman, TX', 'Salisbury, MD', 'Buffalo, NY',
       'Auburn, AL', 'Billings, MT', 'Albany, NY', 'Hinesville, GA',
       'Rome, GA', 'Yuma, AZ', 'Champaign, IL', 'Macon, GA',
       'Evansville, IN', 'New Haven, CT', 'Bend, OR', 'Grand Rapids, MI',
       'Lake Havasu City, AZ', 'Sierra Vista, AZ', 'Santa Maria, CA',
       'Barnstable Town, MA', 'Providence, RI', 'Walla Walla, WA',
       'Lewiston, ID', 'Gainesville, FL', 'Albany, GA',
       'Lexington Park, MD', 'Des Moines, IA', 'Decatur, AL',
       'Tuscaloosa, AL', 'Corvallis, OR', 'Bellingham, WA',
       'Fort Wayne, IN', 'Lafayette, LA', 'Great Falls, MT', 'Utica, NY',
       'Springfield, IL', 'Odessa, TX', 'Saginaw, MI', 'Florence, SC',
       'Manchester, NH', 'Abilene, TX', 'Brownsville, TX',
       'Michigan City, IN', 'Elkhart, IN', 'Waterbury, CT', 'Casper, WY']
    
    prompt = (
        f"Extract the Metropolitan Statistical Area (MSA) from the following US address. "
        f"Map that address to one of the buckets on this list: {msa_list} .\n\nAddress: {address}"
        f"Return only the MSA name (e.g., 'Phoenix, AZ' or 'Washington, DC') not the full address or any other text."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0
        )
        msa = response.choices[0].message.content.strip()
        return msa
    except Exception as e:
        st.error(f"Error extracting MSA via OpenAI: {e}")
        return None

# --- Competitor matching function ---
def find_competitors(msa, subject_property_id, property_data, weighting_list):
    """
    Identify competing properties within a given MSA.
    """
    msa_property_data = property_data[property_data['MSA'] == msa].copy()
    
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
    
    # Get the subject property data
    subject_property = msa_property_data[msa_property_data['PROPERTY_ID'] == subject_property_id].copy()
    
    if subject_property.empty:
        raise ValueError(f"Subject property ID {subject_property_id} not found in MSA {msa}")
    
    # Create comparison data only for the subject property against all other properties
    subject_cols = ['PROPERTY_ID', 'NAME', 'SUBMARKET', 'LONGTITUDE', 'LATITUDE']
    subject_data = subject_property[subject_cols].copy()
    
    # Create a cross join between subject property and all properties in MSA
    result = pd.merge(
        subject_data.assign(key=1),
        msa_property_data.assign(key=1),
        on='key', 
        suffixes=('_S', '')
    ).drop('key', axis=1)
    
    # Rename columns in the comparison data to match expected format
    result = result.rename(columns={
        'PROPERTY_ID': 'PROPERTY_ID_C',
        'NAME': 'NAME_C',
        'SUBMARKET': 'SUBMARKET_C'
    })
    
    # Calculate distance between subject property and all other properties
    result['distance_in_miles'] = result.apply(lambda x: calculate_geospatial_distance(
            loc1=(x['LATITUDE_S'], x['LONGTITUDE_S']),
            loc2=(x['LATITUDE'], x['LONGTITUDE'])
        ), axis=1)
    
    # Filter to properties within 15 miles
    comp_set = result[result['distance_in_miles'] <= 15].reset_index(drop=True)
    
    # Count properties within 1 mile
    one_mile = comp_set[comp_set['distance_in_miles'] <= 1]
    
    # Get the subject property row for reference
    subject_property_row = comp_set[comp_set['PROPERTY_ID_C'] == subject_property_id]
    
    # Filter by walk and bike scores
    for col in ['WS_WALKSCORE', 'WS_BIKESCORE']:
        comp_set[col] = pd.to_numeric(comp_set[col], errors='coerce')
    
    if not subject_property_row.empty:
        subject_walkscore = subject_property_row['WS_WALKSCORE'].values[0]
        subject_bikescore = subject_property_row['WS_BIKESCORE'].values[0]
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
    weighting_array = np.array(weighting_list[:len(training_cols)])
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
                                                     'UNIT_1_MIX', 'UNIT_2_MIX', 'UNIT_3_MIX', 'Latest survey_date', 
                                                     'Integrated', 'Sourced', 'Viable', 'DataQuality', 'Is_Auto_Survey']]
                                                     
    
    competitors['Similarity_Score'] = similarity_score
    competitors['nr_of_props_in_comp_set'] = len(comp_set)
    competitors['nr_of_props_within_a_mile'] = len(one_mile)
    
    output_columns = ['PROPERTY_ID_S', 'NAME_S', 'NAME_C', 'YEAR_BUILT', 'distance_in_miles', 
                      'CONSTRUCTION_TYPE', 'WS_BIKESCORE', 'WS_WALKSCORE', 'UNITS', 
                      'Similarity_Score', 'Latest survey_date', 'Integrated', 'Sourced',  'Viable',
                      'DataQuality', 'Is_Auto_Survey']
    
    
    return competitors[output_columns]

# --- Check if property exists ---
def check_if_property_exists(radix_df, new_row):
    """
    Checks if a new property already exists in the existing properties (radix_df)
    using fuzzy matching on the normalized address and property name.
    """
    best_candidate = None
    best_candidate_tuple = (0, 0)
    # Filter existing properties by state for efficiency.
    candidates = radix_df[radix_df['STATE'] == new_row['State']]
    
    for index, radix_row in candidates.iterrows():
        addr_score = get_similarity_score(radix_row['addr_norm'], new_row['address_norm'])
        name_score = get_similarity_score(radix_row['name_norm'], new_row['name_norm'])
        exact_match = (radix_row['addr_norm'] == new_row['address_norm'])
        cond2 = (addr_score >= 0.60 and name_score >= 0.90)
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
                    'addressScore': addr_score,
                    'nameScore': name_score,
                    'saleforceId': new_row.get('Id', '')  # Preserve the original Id
                }
    return best_candidate

# --- Add new property ---
def get_or_create_property_info(property_data, new_row):
    """
    Checks if the property exists. If not, uses geocoding, WalkScore, and MSA extraction to add it.
    """
    # First check using fuzzy matching against the existing properties.
    # Create a temporary copy of property_data with normalized fields for matching
    temp_property_data = property_data.copy()
    temp_property_data['addr_norm'] = temp_property_data['ADDRESS'].apply(normalize_address)
    temp_property_data['name_norm'] = temp_property_data['NAME'].apply(normalize_property_name)
    
    candidate = check_if_property_exists(temp_property_data, new_row)
    if candidate:
        return property_data, False, candidate  # Not a new property.
    
    # Get coordinates for the new address.
    lat, lon = geocode_address_nominatim(new_row['Address'])
    
    # Get Walk and Bike Scores.
    walkscore, bikescore, _ = get_walk_and_bike_scores(new_row['Address'], lat, lon)
    
    # Get the MSA using OpenAI.
    msa = get_msa_from_address(new_row['Address'])
    
    # Create a new property row.
    new_property_id = np.random.randint(100000, 999999)
    new_property = {
        "PROPERTY_ID": new_property_id,
        "NAME": new_row["Property Name"],
        "ADDRESS": new_row["Address"],
        "MSA": msa,
        "LONGTITUDE": lon,
        "LATITUDE": lat,
        "YEAR_BUILT": new_row.get("YearBuilt", None),
        "CONSTRUCTION_TYPE_NR": 2,  
        "WS_WALKSCORE": walkscore,
        "WS_BIKESCORE": bikescore,
        "UNITS": new_row.get("# Units", None),
        "UNIT_0_MIX": None,
        "UNIT_1_MIX": None,
        "UNIT_2_MIX": None,
        "UNIT_3_MIX": None,
        "SQFT_0": None,
        "SQFT_1": None,
        "SQFT_2": None,
        "SQFT_3": None,
    }
    
    new_row_df = pd.DataFrame([new_property])
    property_data = pd.concat([property_data, new_row_df], ignore_index=True)
    return property_data, True, new_property

# --- Define weighting list ---
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

weighting_list = [weight_distance_in_miles, weight_year_build, weight_construction_type, weight_ws_bikescore, 
                 weight_ws_walkscore, weight_units, weight_unit_mix_0, weight_unit_mix_1, weight_unit_mix_2, 
                 weight_unit_mix_3, weight_sqft_0, weight_sqft_1, weight_sqft_2, weight_sqft_3]

# --- Load static data ---
@st.cache_data
def load_static_data():
    radix_file = 'data_preperation_for_comp_match_1.csv'
    # Both property_data and radix_df are from the same static source.
    radix_df = pd.read_csv(radix_file, dtype=str)
    property_data = pd.read_csv(radix_file, dtype=str)
    property_data = property_data.copy()  # This ensures all columns are loaded

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

# --- Workflow functions ---
def run_radix_workflow(mapping_df, property_data, weighting_list):
    """Run competitor analysis for properties already in Radix."""
    st.header("Competitor Analysis for Properties in Radix")
    
    # Summary statistics
    st.write(f"Processing {len(mapping_df)} properties that match existing Radix data")
    st.info("Note: For some properties, we might not have good competitors to recommend based on available data.")
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Fun messages to display while processing
    fun_messages = [
        "Finding the perfect competitors...",
        "Crunching the numbers like a pro...",
        "You're doing great! Just a bit longer...",
        "This data is looking fantastic!",
        "You have excellent taste in property data!",
        "Making magic happen with your data...",
        "Almost there! Your results will be worth the wait...",
        "You're going to love these insights!",
        "Connecting the dots between properties..."
    ]
    
    competitor_results_list = []
    errors_count = 0
    
    for idx, row in enumerate(mapping_df.iterrows()):
        # Update progress
        progress = (idx + 1) / len(mapping_df)
        progress_bar.progress(progress)
        
        # Display a random encouraging message
        if idx % 2 == 0:
            status_text.write(np.random.choice(fun_messages))
        
        # Process the property
        _, row_data = row
        subject_property_id = row_data['RadixPropertyID']
        msa = row_data['MSA']
        original_id = row_data.get('OriginalId', '')
        
        try:
            comp_df = find_competitors(msa, subject_property_id, property_data, weighting_list)
            
            # Add the original Id to all rows in the competitor results
            if original_id:
                comp_df['OriginalId'] = original_id
                
            competitor_results_list.append(comp_df)
        except Exception:
            # Silently count errors without displaying them
            errors_count += 1
            continue
    
    # Clear the status text and show completion
    status_text.empty()
    progress_bar.empty()
    
    # Combine all results into a single DataFrame
    if competitor_results_list:
        final_competitors_df = pd.concat(competitor_results_list, ignore_index=True)
        csv_data = final_competitors_df.to_csv(index=False).encode("utf-8")
        
        st.success(f"✅ Analysis complete! Found {len(final_competitors_df)} competitor matches.")
        if errors_count > 0:
            st.info(f"Note: {errors_count} properties couldn't be matched with good competitors.")
        
        st.download_button(
            label="Download Competitor Results CSV",
            data=csv_data,
            file_name="competitor_results_radix.csv",
            mime="text/csv",
            help="Click to download the complete competitor analysis results"
        )
    else:
        st.warning("No competitor results generated. Try adjusting your criteria or adding more properties.")

def run_new_property_workflow(new_df_not_in_radix, property_data, weighting_list):
    """Add new properties and run competitor analysis on them."""
    st.header("Adding New Properties and Running Competitor Analysis")
    
    # Summary statistics
    st.write(f"Processing {len(new_df_not_in_radix)} new properties not in Radix")
    st.info("Note: For some properties, we might not have good competitors to recommend based on available data.")
    
    # Create progress indicators
    add_progress = st.progress(0)
    add_status = st.empty()
    
    # Fun messages for adding properties
    add_messages = [
        "Adding this beautiful property to our database...",
        "This property looks amazing! Adding it now...",
        "Expanding our property universe...",
        "Another great addition to the collection!",
        "You've found some excellent properties!"
    ]
    
    newly_added_properties = []
    for idx, row in enumerate(new_df_not_in_radix.iterrows()):
        # Update progress
        progress = (idx + 1) / len(new_df_not_in_radix)
        add_progress.progress(progress)
        
        # Display a random encouraging message
        if idx % 2 == 0:
            add_status.write(np.random.choice(add_messages))
        
        # Process the property
        _, row_data = row
        original_id = row_data.get('Id', '')
        
        try:
            property_data, was_new, property_info = get_or_create_property_info(property_data, row_data)
            if was_new:
                # Add the original Id to the property info
                property_info['OriginalId'] = original_id
                newly_added_properties.append(property_info)
        except Exception:
            # Silently continue if there's an error
            continue
    
    # Clear the status and progress
    add_status.empty()
    add_progress.empty()
    
    if not newly_added_properties:
        st.warning("No new properties were added. Check your data and try again.")
        return
    
    st.success(f"✅ Successfully added {len(newly_added_properties)} new properties!")
    
    # Now run competitor analysis on the new properties
    st.subheader("Running Competitor Analysis")
    
    # Create new progress indicators
    comp_progress = st.progress(0)
    comp_status = st.empty()
    
    # Fun messages for competitor analysis
    comp_messages = [
        "Finding the perfect competitors...",
        "Matching properties like a dating app, but for buildings...",
        "Your data is in good hands!",
        "Analyzing neighborhood dynamics...",
        "This is where the magic happens..."
    ]
    
    competitor_results_list = []
    errors_count = 0
    
    for idx, prop in enumerate(newly_added_properties):
        # Update progress
        progress = (idx + 1) / len(newly_added_properties)
        comp_progress.progress(progress)
        
        # Display a random encouraging message
        if idx % 2 == 0:
            comp_status.write(np.random.choice(comp_messages))
        
        subject_property_id = prop["PROPERTY_ID"]
        msa = prop["MSA"]
        original_id = prop.get("OriginalId", "")
        
        try:
            comp_df = find_competitors(msa, subject_property_id, property_data, weighting_list)
            
            # Add the original Id to all rows in the competitor results
            if original_id:
                comp_df['OriginalId'] = original_id
                
            competitor_results_list.append(comp_df)
        except Exception:
            # Silently count errors without displaying them
            errors_count += 1
    
    # Clear the status and progress
    comp_status.empty()
    comp_progress.empty()
    
    # Combine all results into a single DataFrame
    if competitor_results_list:
        final_competitors_df = pd.concat(competitor_results_list, ignore_index=True)
        csv_data = final_competitors_df.to_csv(index=False).encode("utf-8")
        
        st.success(f"✅ Analysis complete! Found {len(final_competitors_df)} competitor matches.")
        if errors_count > 0:
            st.info(f"Note: {errors_count} properties couldn't be matched with good competitors.")
        
        st.download_button(
            label="Download Competitor Results CSV",
            data=csv_data,
            file_name="competitor_results_new_properties.csv",
            mime="text/csv",
            help="Click to download the complete competitor analysis results"
        )
    else:
        st.warning("No competitor results generated. Try adjusting your criteria or adding more properties.")

# --- Main application logic ---
radix_df, property_data = load_static_data()

st.sidebar.header("Upload New Properties CSV")
uploaded_new_file = st.sidebar.file_uploader("Upload CSV", type="csv")

if uploaded_new_file is not None:
    try:
        new_df = pd.read_csv(uploaded_new_file, dtype=str)
        
        # Check if the file has the new format (with Property_Name__c, Units__c, etc.)
        if 'Property_Name__c' in new_df.columns:
            st.sidebar.info("Detected new file format. Converting columns...")
            
            # Create a mapping dictionary for column renaming
            column_mapping = {
                'Name': 'Name',
                'Property_Name__c': 'Property Name',
                'Units__c': '# Units',
                'Street_Address__c': 'Street Address',
                'City__c': 'City',
                'State__c': 'State',
                'ZIP_Code__c': 'ZIP'
            }
            
            # Rename columns based on the mapping (only for columns that exist)
            rename_cols = {old: new for old, new in column_mapping.items() if old in new_df.columns}
            new_df = new_df.rename(columns=rename_cols)
            
            # Combine address components if they're not already combined
            if 'Address' not in new_df.columns and 'Street Address' in new_df.columns:
                new_df['Address'] = new_df.apply(
                    lambda row: f"{row.get('Street Address', '')}, {row.get('City', '')}, {row.get('State', '')}, {row.get('ZIP', '')}",
                    axis=1
                )
                st.sidebar.success("Successfully combined address fields")
        
        # Normalize new_df properties
        new_df['address_norm'] = new_df['Address'].apply(normalize_address)
        new_df['name_norm'] = new_df['Property Name'].apply(normalize_property_name)
        new_df['State'] = new_df['Address'].apply(extract_state)
        
        st.sidebar.success(f"Successfully loaded file with {len(new_df)} properties")
        
        # --- Mapping: Fuzzy Matching from Radix to New ---
        st.header("Property Matching Analysis")
        
        mapping_results = []
        new_properties = []
        
        for index, new_row in new_df.iterrows():
            candidate = check_if_property_exists(radix_df, new_row)
            if candidate:
                mapping_results.append(candidate)
            else:
                new_properties.append(new_row)
        
        mapping_df = pd.DataFrame(mapping_results)
        new_df_not_in_radix = pd.DataFrame(new_properties)
        
        st.write(f"Total properties in uploaded file: {len(new_df)}")
        st.write(f"Properties matched to existing Radix data: {len(mapping_df)}")
        st.write(f"New properties not in Radix: {len(new_df_not_in_radix)}")
        
        if not mapping_df.empty:
            st.subheader("Matched Properties")
            st.dataframe(mapping_df)
                
        # --- Workflow Selection ---
        st.sidebar.header("Select Workflow")
        workflow = st.sidebar.radio(
            "Choose which properties to analyze:",
            ["Properties in Radix", "New Properties (Not in Radix)"]
        )
        
        if workflow == "Properties in Radix":
            if mapping_df.empty:
                st.warning("No properties matched with Radix database. Please select 'New Properties' workflow instead.")
            else:
                run_radix_workflow(mapping_df, property_data, weighting_list)
        else:  # New Properties workflow
            if new_df_not_in_radix.empty:
                st.warning("No new properties found. All properties in your file already exist in Radix.")
            else:
                run_new_property_workflow(new_df_not_in_radix, property_data, weighting_list)
        
    except Exception as e:
        st.error(f"Error reading or processing CSV: {e}")
        st.stop()
else:
    st.info("Please upload a new properties CSV file to proceed.")
    st.stop()
