from flask_migrate import Migrate
from flask_wtf.csrf import CSRFError
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from flask_wtf import CSRFProtect
from models import Officer, Admin
from models import extract_features_from_text
from models import db, Officer, Admin
from forms import OfficerEditForm 
from forms import PredictCrimeForm 
from forms import AdminLoginForm
from forms import HeatmapFilterForm
from forms import UploadEvidenceForm
from forms import EmptyForm 
from forms import (AdminRegistrationForm, OfficerRegistrationForm, AdminResetPasswordForm)
from forms import ParagraphForm
from utils.nlp_parser import extract_features_from_text
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from heatmap import generate_crime_bubble_map 
from blockchain import Blockchain 
from encrypt import encrypt_file, decrypt_file
import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import json
import plotly
import random
import plotly.graph_objects as go
import pickle

# =========================== App Configuration ===========================
load_dotenv()
app = Flask(__name__)

blockchain = Blockchain() 
blockchain.add_block("Your forensic evidence data here")

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY') or 'your-very-secret-key'
csrf = CSRFProtect(app)

@app.errorhandler(CSRFError)
def handle_csrf_error(e):
    flash('CSRF token missing or invalid. Please try again.', 'danger')
    return redirect(request.url)

basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'instance', 'users.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db.init_app(app)
migrate = Migrate(app, db)

# =========================== Login Manager ===========================
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return Officer.query.get(int(user_id)) or Admin.query.get(int(user_id))

# =========================== Load ML Models ===========================
# ✅ Load the correct files
model = joblib.load('best_model.joblib')
scaler = joblib.load('scaler.joblib')
label_encoders = joblib.load('label_encoders.joblib')

# Load encoders
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

print("🔍 Available label encoder keys:", label_encoders.keys())  # ADD THIS LINE

# =========================== Routes ===========================

@app.route('/')
def index():
    return render_template('index.html')

# --------------------------- Officer Panel ---------------------------

@app.route('/officer_register', methods=['GET', 'POST'])
def officer_register():
    form = OfficerRegistrationForm()
    if form.validate_on_submit():
        new_officer = Officer(
            full_name=form.full_name.data,
            email=form.email.data,
            phone=form.phone.data,
            rank=form.rank.data,
            department=form.department.data,
            badge_id=form.badge_id.data,
            username=form.username.data,
            password=generate_password_hash(form.password.data)
        )
        db.session.add(new_officer)
        db.session.commit()
        flash('Registration successful! Awaiting admin approval.', 'success')
        return redirect(url_for('login'))  # Replace 'login' if you use a different login endpoint
    return render_template('officer_register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = Officer.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            if user.is_approved:
                login_user(user)
                flash('Login successful.', 'success')
                return redirect(url_for('index'))
            else:
                flash('Registration submitted. Waiting for admin approval.', 'warning')
                return redirect(url_for('login'))
        else:
            flash('Invalid credentials.', 'danger')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("Logged out successfully.", "info")
    return redirect(url_for('index'))

# --------------------------- Crime Prediction ---------------------------

@app.route('/predict', methods=['GET'])
def predict():
    """
    Main prediction interface with both Form and Paragraph options.
    """
    return render_template('predict.html')

@app.route('/predict_from_form', methods=['GET', 'POST'])
def predict_from_form():
    form = PredictCrimeForm()
    # Load encoders for dropdowns
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    # Populate SelectField choices dynamically
    form.location_type.choices = [(val, val) for val in label_encoders['location_type'].classes_]
    form.time_of_day.choices = [(val, val) for val in label_encoders['time_of_day'].classes_]
    form.day_of_week.choices = [(val, val) for val in label_encoders['day_of_week'].classes_]
    form.weapon_involved.choices = [(val, val) for val in label_encoders['weapon_involved'].classes_]
    form.known_offender.choices = [(val, val) for val in label_encoders['known_offender'].classes_]
    form.prior_incidents.choices = [(val, val) for val in label_encoders['prior_incidents'].classes_]
    if form.validate_on_submit():
        try:
            # Encode categorical inputs
            encoded_data = [
                label_encoders['location_type'].transform([form.location_type.data])[0],
                label_encoders['time_of_day'].transform([form.time_of_day.data])[0],
                label_encoders['day_of_week'].transform([form.day_of_week.data])[0],
                label_encoders['weapon_involved'].transform([form.weapon_involved.data])[0],
                label_encoders['known_offender'].transform([form.known_offender.data])[0],
                label_encoders['prior_incidents'].transform([form.prior_incidents.data])[0],
                form.num_suspects.data
            ]

            # Predict
            model = joblib.load('crime_model.pkl')
            prediction_encoded = model.predict([encoded_data])[0]
            prediction = label_encoders['crime_type'].inverse_transform([prediction_encoded])[0]

            features = {
                'location_type': form.location_type.data,
                'time_of_day': form.time_of_day.data,
                'day_of_week': form.day_of_week.data,
                'weapon_involved': form.weapon_involved.data,
                'known_offender': form.known_offender.data,
                'prior_incidents': form.prior_incidents.data,
                'num_suspects': form.num_suspects.data
            }
            return render_template('predict_result.html', prediction=prediction, features=features)
        except Exception as e:
            return f"❌ Error during prediction: {e}"
    else:
        # Optional: Debug validation
        print(form.errors)
    return render_template('predict_form.html', form=form)

@app.route('/predict_from_text', methods=['GET', 'POST'])
@login_required
def predict_from_text():
    form = ParagraphForm()
    if form.validate_on_submit():
        paragraph = form.crime_paragraph.data
        # Extract features from the paragraph
        features = extract_features_from_text(paragraph)
        # Load model and encoders
        model = joblib.load('crime_model.pkl')
        with open('label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        # Encode features
        encoded_features = []
        for key in ['location_type', 'time_of_day', 'day_of_week', 'weapon_involved', 'known_offender', 'prior_incidents']:
            value = features[key]
            le = label_encoders[key]

            if value in le.classes_:
                encoded = le.transform([value])[0]
            else:
                encoded = le.transform([le.classes_[0]])[0]

            encoded_features.append(encoded)
        # Append numeric input
        encoded_features.append(features['num_suspects'])
        # Predict
        prediction = model.predict([encoded_features])[0]
        crime_type = label_encoders['crime_type'].inverse_transform([prediction])[0]
        return render_template('predict_result.html', paragraph=paragraph, crime_type=crime_type)
    return render_template('predict_text.html', form=form)

# --------------------------- Upload & Files ---------------------------

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    form = UploadEvidenceForm()
    if form.validate_on_submit():
     file = form.file.data
     filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
     file.save(filepath)
     encrypt_file(filepath)
     blockchain.add_block(data=f"Uploaded: {file.filename}")  # <-- add this line
     flash('File encrypted, saved, and recorded on blockchain!', 'success')
     return redirect(url_for('upload'))
    return render_template('upload.html', form=form)

@app.route('/files')
@login_required
def list_files():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    form = EmptyForm()
    return render_template('files.html', files=files, form=form)


@app.route('/uploads/<filename>')
@login_required
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/delete/<filename>', methods=['POST'])
@login_required
def delete_file(filename):
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.remove(filepath)
        flash(f'File \"{filename}\" deleted.', 'success')
    except Exception as e:
        flash(f'Error deleting file: {e}', 'danger')
    return redirect(url_for('list_files'))

# --------------------------- Admin Panel ---------------------------
@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    if current_user.__class__ != Admin:
        flash("Unauthorized access", "danger")
        return redirect(url_for('index'))
    # Welcoming the admin
    admin_name = current_user.full_name
    # Fetch pending users for approval
    pending_users = Officer.query.filter_by(is_approved=False).all()
    return render_template(
        'admin_dashboard.html',
        admin_name=admin_name,
        pending_users=pending_users
    )

@app.route('/admin/users')
@login_required
def admin_users():
    if current_user.__class__ != Admin:
        flash("Unauthorized access", "danger")
        return redirect(url_for('index'))
    officers = Officer.query.all()
    return render_template('admin_users.html', officers=officers)

@app.route('/admin/reset-password/<int:user_id>', methods=['GET', 'POST'])
@login_required
def admin_reset_password(user_id):
    if current_user.__class__ != Admin:
        flash("Access denied.", "danger")
        return redirect(url_for('index'))
    form = AdminResetPasswordForm()
    user = Officer.query.get_or_404(user_id)
    if form.validate_on_submit():
        user.password = generate_password_hash(form.new_password.data)
        db.session.commit()
        flash(f"Password reset for Officer {user.full_name}.", "success")
        return redirect(url_for('admin_users'))
    return render_template('admin_reset_password.html', form=form, user=user)

@app.route('/admin/decrypt', methods=['GET', 'POST'])
@login_required
def admin_decrypt():
    if current_user.__class__ != Admin:
        flash("Access denied.", "danger")
        return redirect(url_for('index'))
    decrypted_content, selected_file = None, None
    files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith('.csv')]
    if request.method == 'POST':
        selected_file = request.form.get('file')
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], selected_file)
        output = os.path.join(app.config['UPLOAD_FOLDER'], 'decrypted_' + selected_file)
        try:
            decrypt_file(filepath, output)
            with open(output, 'r') as f:
                decrypted_content = f.read()
            os.remove(output)
        except Exception as e:
            flash(f"Decryption failed: {str(e)}", "danger")
    return render_template('admin_decrypt.html', files=files, content=decrypted_content, selected_file=selected_file)

@app.route('/admin/register', methods=['GET', 'POST'])
def admin_register():
    form = AdminRegistrationForm()
    if form.validate_on_submit():
        new_admin = Admin(
            full_name=form.full_name.data,
            email=form.email.data,
            username=form.username.data,
            password=generate_password_hash(form.password.data)
        )
        db.session.add(new_admin)
        db.session.commit()
        flash('Admin registered successfully.', 'success')
        return redirect(url_for('admin_login'))
    return render_template('admin_register.html', form=form)

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    form = AdminLoginForm() 
    if request.method == 'POST' and form.validate_on_submit():
        username = request.form['username']
        password = request.form['password']
        admin = Admin.query.filter_by(username=username).first()

        if admin and check_password_hash(admin.password, password):
            login_user(admin)  # This will log the admin in
            flash('Welcome Admin!', 'success')
            return redirect(url_for('admin_dashboard'))  # Redirect to admin dashboard after login
        else:
            flash('Invalid admin credentials.', 'danger')
    return render_template('admin_login.html',form=form)  # Renders the login page

@app.route('/admin/pending_approvals')
@login_required
def pending_officer_approvals():
    if current_user.__class__ != Admin:
        flash("Unauthorized access", "danger")
        return redirect(url_for('index'))

    pending_users = Officer.query.filter_by(is_approved=False).all()
    return render_template('admin_pending_approvals.html', pending_users=pending_users)

@app.route('/admin/approve/<int:user_id>', methods=['POST'])
@login_required
def approve_user(user_id):
    if current_user.__class__ != Admin:
        flash("Unauthorized access", "danger")
        return redirect(url_for('index'))

    officer = Officer.query.get_or_404(user_id)
    officer.is_approved = True
    db.session.commit()
    flash('Officer approved successfully!', 'success')
    return redirect(url_for('pending_officer_approvals'))

@app.route('/admin/disapprove/<int:officer_id>')
@login_required
def disapprove_officer(officer_id):
    if current_user.__class__ != Admin:
        flash("Unauthorized access", "danger")
        return redirect(url_for('index'))

    officer = Officer.query.get_or_404(officer_id)
    db.session.delete(officer)
    db.session.commit()
    flash('Officer disapproved and removed.', 'warning')
    return redirect(url_for('pending_officer_approvals'))

@app.route('/admin/edit_officer/<int:officer_id>', methods=['GET', 'POST'])
def officer_edit(officer_id):
    officer = Officer.query.get_or_404(officer_id)
    form = OfficerEditForm(obj=officer)

    if form.validate_on_submit():
        officer.full_name = form.full_name.data
        officer.username = form.username.data
        officer.phone = form.phone.data
        officer.rank = form.rank.data
        officer.department = form.department.data
        officer.email = form.email.data
        officer.badge_id = form.badge_id.data
        db.session.commit()
        flash('Officer updated successfully!', 'success')
        return redirect(url_for('manage_officers'))

    return render_template('officer_edit.html', form=form, officer=officer)

@app.route('/admin/delete_officer/<int:officer_id>')
@login_required
def delete_officer(officer_id):
    if current_user.__class__ != Admin:
        flash("Unauthorized access", "danger")
        return redirect(url_for('index'))

    officer = Officer.query.get_or_404(officer_id)
    db.session.delete(officer)
    db.session.commit()
    flash('Officer deleted successfully!', 'success')
    return redirect(url_for('manage_officers'))

@app.route('/admin/manage_officers')
@login_required
def manage_officers():
    if current_user.__class__ != Admin:
        flash("Unauthorized access", "danger")
        return redirect(url_for('index'))

    officers = Officer.query.filter_by(is_approved=True).all()
    return render_template('manage_officers.html', officers=officers)

# --------------------------- Charts & Stats ---------------------------

@app.route("/filter_form", methods=["GET"])
@login_required
def filter_form():
    # Load dataset for options
    df = pd.read_csv("data/crime_data.csv")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['Year'] = df['date'].dt.year

    # Unique dropdown options
    unique_years = sorted(df['Year'].dropna().unique().astype(int), reverse=True)
    unique_crime_types = sorted(df['crime_type'].dropna().unique())
    unique_locations = sorted(df['location_type'].dropna().unique())

    return render_template(
        "filter_form.html",
        years=unique_years,
        crime_types=unique_crime_types,
        locations=unique_locations
    )

@app.route("/filtered_data", methods=["GET"])
@login_required
def filtered_data():
    year = request.args.get('year')
    crime_type = request.args.get('crime_type')
    location = request.args.get('location')
    print(f"[DEBUG] Filter parameters received\nYear: {year}, Crime Type: {crime_type}, Location: {location}")

    # Load dataset
    df = pd.read_csv("data/crime_data.csv")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['Year'] = df['date'].dt.year
    df['Month'] = df['date'].dt.month

    # Apply filters
    filtered_df = df.copy()
    if year:
        filtered_df = filtered_df[filtered_df['Year'].astype(str) == str(year)]
    if crime_type:
        filtered_df = filtered_df[filtered_df['crime_type'].str.lower() == crime_type.lower()]
    if location:
        filtered_df = filtered_df[filtered_df['location_type'].str.lower() == location.lower()]

    print("[DEBUG] Filtered DataFrame shape:", filtered_df.shape)

    return render_template(
        "filtered_data.html",
        filtered_df=filtered_df,
        year=year,
        crime_type=crime_type,
        location=location
    )

@app.route('/crime_heatmap', methods=['GET', 'POST'])
@login_required
def crime_heatmap():
    form = HeatmapFilterForm()
    df = pd.read_csv('data/crime_data.csv')

    # populate choices
    form.time_of_day.choices = [('', 'All')] + [(t, t) for t in sorted(df['time_of_day'].unique())]
    form.location_type.choices = [('', 'All')] + [(l, l) for l in sorted(df['location_type'].unique())]

    # on POST grab selected values
    if form.validate_on_submit():
        selected_time = form.time_of_day.data
        selected_location = form.location_type.data
    else:
        selected_time = None
        selected_location = None

    graphJSON = generate_crime_bubble_map(df, selected_time, selected_location)
    return render_template(
        'crime_heatmap.html',
        form=form,
        graphJSON=graphJSON
    )

@app.route('/crime_trends')
@login_required
def crime_trends():
    # Load your dataset
    df = pd.read_csv('data/crime_data.csv')
    print("CSV columns:", df.columns)  # Debug: check what you have

    if 'date' not in df.columns:
        return f"CSV missing required column 'date'. Found columns: {list(df.columns)}", 400

    # Convert 'date' to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)

    # Extract year
    df['year'] = df['date'].dt.year

    # Count number of crimes per year
    grouped = df.groupby('year').size().reset_index(name='crime_count')
    grouped = grouped.sort_values('year')

    print("Prepared data for plot:\n", grouped)

    years = grouped['year'].tolist()
    crime_counts = grouped['crime_count'].tolist()

    # Create styled bar chart with Plotly
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=years,
        y=crime_counts,
        marker=dict(
            color='crimson',
            line=dict(color='white', width=1),
        ),
        hoverinfo='x+y',
    ))

    fig.update_layout(
    title={
        # 'text': 'Crime Trends Over the Years',
        'x': 0.5,
        'xanchor': 'center',
        'font': dict(size=24, color='white'),
    },
    xaxis=dict(
        title=dict(
            text='Year',
            font=dict(color='white', size=18),
        ),
        gridcolor='gray',
        linecolor='white',
        tickfont=dict(color='white'),
    ),
    yaxis=dict(
        title=dict(
            text='Number of Crimes',
            font=dict(color='white', size=18),
        ),
        gridcolor='gray',
        linecolor='white',
        tickfont=dict(color='white'),
    ),
    plot_bgcolor='#1e1e2f',
    paper_bgcolor='#1e1e2f',
    font=dict(color='white', size=16),
    hovermode='x unified',
)

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('crime_trends.html', graphJSON=graphJSON)

@app.route('/top_locations')
def top_locations():
    # Load CSV fresh every time
    df = pd.read_csv('data/crime_data.csv')
    df['location_type'] = df['location_type'].fillna('Unknown')
    
    # Compute top 10
    top_locations = df['location_type'].value_counts().nlargest(10).reset_index()
    top_locations.columns = ['Location', 'Counts']

    print("\n==========[DEBUG: Top 10 locations DataFrame]==========")
    print(top_locations)
    print("=======================================================\n")

    # Correctly build the pie chart with real counts
    fig = px.pie(
        top_locations,
        names="Location",
        values="Counts",  # must point to actual counts column
        title="Top 10 Crime Locations - Pie Chart",
        hole=0.4
    )

    fig.update_traces(
        textinfo='label+percent',
        marker=dict(line=dict(color='#000000', width=2))
    )

    fig.update_layout(
        paper_bgcolor="#14141b",
        plot_bgcolor="#14141b",
        font_color="#FFFFFF",
        title_x=0.5,
    )

    pie_chart_html = fig.to_html(full_html=False)

    return render_template(
        "top_locations.html",
        pie_chart=pie_chart_html
    )

# --------------------------- Crime Dashboard ---------------------------

@app.route('/crime_dashboard')
def crime_dashboard():
    # Load data
    df = pd.read_csv('data/crime_data.csv')

    # 1) Pie chart of top crime locations
    location_counts = df['location_type'].value_counts().head(10)
    fig_pie = px.pie(
        location_counts,
        names=location_counts.index,
        values=location_counts.values,
        hole=0.3,
        title="Top Crime Locations - Pie Chart",
    )
    pie_html = fig_pie.to_html(full_html=False)

    # 2) Bar chart of top crime types
    crime_counts = df['crime_type'].value_counts().head(10)
    fig_bar = px.bar(
        crime_counts,
        x=crime_counts.index,
        y=crime_counts.values,
        color=crime_counts.index,
        title="Top Crime Types - Bar Chart",
        labels={'x': 'Crime Type', 'y': 'Count'},
    )
    bar_html = fig_bar.to_html(full_html=False)

    # 3) Crime heatmap
    fig_map = px.density_mapbox(
        df, lat="latitude", lon="longitude", z=None,
        radius=10, center=dict(lat=28.61, lon=77.20), zoom=10,
        mapbox_style="carto-positron",
        title="Crime Heatmap"
    )
    map_html = fig_map.to_html(full_html=False)

    # Pass all charts/maps to template
    return render_template(
        'crime_dashboard.html',
        pie_html=pie_html,
        bar_html=bar_html,
        map_html=map_html,
    )

# --------------------------- Error Pages ---------------------------

@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404

@app.errorhandler(403)
def forbidden(e):
    return render_template("403.html"), 403

# --------------------------- View chain ---------------------------
@app.route('/view_chain')
@login_required
def view_chain():
    blockchain_chain = blockchain.get_full_chain()  # returns list of blocks as dicts
    return render_template('view_chain.html', blockchain=blockchain_chain)
 
@app.route('/verify_chain')
@login_required
def verify_chain():
    is_valid = blockchain.verify_integrity()  # returns True/False
    return render_template('verify_chain.html', is_valid=is_valid)


# =========================== Main ===========================

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
