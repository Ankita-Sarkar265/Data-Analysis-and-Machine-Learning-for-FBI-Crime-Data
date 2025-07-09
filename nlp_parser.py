import re

# Define known values for encoding
known_locations = ['residential', 'commercial', 'public', 'street']
known_times = ['morning', 'afternoon', 'evening', 'night']
known_days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
known_weapons = ['gun', 'knife', 'bat', 'none']

def extract_features_from_text(text):
    text = text.lower()

    # Extract location
    location_type = next((loc for loc in known_locations if loc in text), 'residential')
    
    # Extract time of day
    time_of_day = next((t for t in known_times if t in text), 'evening')

    # Extract day of week
    day_of_week = next((d for d in known_days if d in text), 'friday')

    # Extract number of suspects
    suspects_match = re.search(r'(\d+)\s+suspect', text)
    num_suspects = int(suspects_match.group(1)) if suspects_match else 1

    # Extract weapon involved
    weapon_involved = next((w for w in known_weapons if w in text), 'none')

    # Known offender
    known_offender = 'yes' if 'known offender' in text else 'no'

    # Prior incidents
    prior_incidents_match = re.search(r'(\d+)\s+prior', text)
    prior_incidents = str(prior_incidents_match.group(1)) if prior_incidents_match else '0'

    return {
        'location_type': location_type,
        'time_of_day': time_of_day,
        'day_of_week': day_of_week,
        'num_suspects': num_suspects,
        'weapon_involved': weapon_involved,
        'known_offender': known_offender,
        'prior_incidents': prior_incidents
    }
