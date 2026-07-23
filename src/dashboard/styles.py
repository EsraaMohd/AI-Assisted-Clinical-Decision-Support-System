CUSTOM_CSS = """
<style>

/* Hide Streamlit Menu */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}


/* Background */

.stApp{

    background-color:#f8fafc;

}


/* Main Title */

h1{

    color:#0f172a;

    font-weight:700;

}


/* Section Titles */

h2,h3{

    color:#1e3a8a;

}


/* Metric Cards */

div[data-testid="metric-container"]{

    background:white;

    border-radius:15px;

    border:1px solid #dbe4ee;

    padding:18px;

    box-shadow:0px 3px 8px rgba(0,0,0,.08);

}


/* Buttons */

.stButton>button{

    width:100%;

    height:55px;

    font-size:18px;

    font-weight:bold;

    border-radius:12px;

}


/* Sidebar */

section[data-testid="stSidebar"]{

    background:#eef5fb;

}


/* Expander */

.streamlit-expanderHeader{

    font-size:18px;

    font-weight:bold;

}

</style>
"""