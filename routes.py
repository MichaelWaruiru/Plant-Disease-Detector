import os
import logging
from flask import render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from app import app
# from ml_model import PlantDiseaseModel
from model_instance import model
from disease_data import get_disease_info, get_treatment_recommendations

# Initialize the model
# model = PlantDiseaseModel()

# Allowed file extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}

def allowed_file(filename):
  """Check if file has allowed extension"""
  return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
  return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
  """Handle file upload and disease detection"""
  try:
    # Check if file was uploaded
    if "file" not in request.files:
      flash("No file selected", "error")
      return redirect(url_for("index"))
    
    file = request.files["file"]
    
    # Check if file was selected
    if file.filename == "":
      flash("No file selected", "error")
      return redirect(url_for("index"))
    
    # Check if file type is allowed
    if not allowed_file(file.filename):
      flash("Invalid file type. Please upload an image file(PNG, JPG, JPEG, GIF, BMP, WEBP)", "error")
      return redirect(url_for("index"))
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)
    
    # Perform disease detection
    predictions = model.predict(file_path)
    
    # Get disease information and treatment recommendations
    results = []
    for pred in predictions:
      disease_name = pred["class"]
      confidence = pred["confidence"]
      
      disease_info = get_disease_info(disease_name)
      treatments = get_treatment_recommendations(disease_name)
      
      results.append({
        "disease": disease_name,
        "confidence": confidence,
        "info": disease_info,
        "treatments": treatments
      })
      
      # Clean up the uploaded file
      if os.path.exists(file_path):
        os.remove(file_path)
      
    return render_template("results.html", results=results, filename=filename)
    
  except Exception as e:
    logging.error(f"Error processing upload: {str(e)}")
    flash(f"Error processing image: {str(e)}", "error")
    return redirect(url_for("index"))
  

@app.errorhandler(413)
def too_large(e):
  """Handle file too large error"""
  flash("FIle is too large. Maximum size is 16MB.", "error")
  return redirect(url_for("index"))

@app.errorhandler(404)
def not_found(e):
  """Handle 404 errors"""
  return render_template("404.html"), 404

@app.errorhandler(500)
def internal_error(e):
  """Handle server errors"""
  logging.error(f"Internal server error: {str(e)}")
  return render_template("500.html"), 500
