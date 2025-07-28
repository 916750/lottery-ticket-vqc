FILE_EXTENSIONS = ['pdf', 'png']

color_mapping = {
    "100": "#1f77b4",  # Blue
    "80": "#d62728",   # Red
    "60": "#ff7f0e",   # Orange
    "50": "#2ca02c",   # Green
    "40": "#9467bd",   # Purple
    "30": "#f15c79",   # Magenta
    "26": "#bcbd22",   # Olive
    "20": "#5c9ead",   # Teal
    "16": "#e377c2",   # Pink
    "13": "#8c564b",   # Brown
    "10": "#17becf",   # Cyan
    "8": "gray",
    "3": "gray",
    "1": "orange"
}

# Function to categorize the values
def categorize_value(value):
    value = float(value)
    if value >= 100:
        return "100"
    elif value >= 79:
        return "80"
    elif value >= 60:
        return "60"
    elif value >= 50:
        return "50"
    elif value >= 40:
        return "40"
    elif value >= 30:
        return "30"
    elif value >= 26:
        return "26"
    elif value >= 20:
        return "20"
    elif value >= 16:
        return "16"
    elif value >= 13:
        return "13"
    elif value >= 10:
        return "10"
    elif value >= 8:
        return "8"
    elif value >= 3:
        return "3"
    else:
        return "1"
