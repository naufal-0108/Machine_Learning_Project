import os, numpy as np, io
import torch, cv2
import logging
from time import time
from flask import Flask, request, render_template, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
from utils import load_checkpoint
from model import AttentionSpatialChannelUnetWithDS, Unet


logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler('app.log'),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
MODELS_FOLDER = 'models'
GPU = False

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MODELS_FOLDER'] = MODELS_FOLDER

MODEL_CONFIGS = {
    'attention_unet': {
        "name": "Attention U-Net",
        "params": {
            "num_classes": 1,
            "kernel_size": 3,
            "padding": 1,
            "dropout": 0.1
        },
        "checkpoint": os.path.join(MODELS_FOLDER, 'attention_unet.pt')
    },
    'basic_unet': {
        "name": "Basic U-Net",
        "params": {
            "num_classes": 1,
            "dropout": 0
        },
        "checkpoint": os.path.join(MODELS_FOLDER, 'basic_unet.pt')
    }
}

CURRENT_MODEL = {
    'key': None,
    'model': None
}


def preprocessing(image_dir):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], image_dir)
    try:
        image = np.array(Image.open(filepath).resize((512, 512), resample=Image.BILINEAR))
        image = image/255.
        return image
    except Exception as e:
        logger.error(f"Error preprocessing image {image_dir}: {e}")
        raise


def load_model(parameters, model_name, checkpoint_dir, gpu=GPU):
    try:
        if model_name == 'attention_unet':
            model = AttentionSpatialChannelUnetWithDS(**parameters)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.0025)
            model, opt, epoch, loss, dice = load_checkpoint(model, optimizer, scheduler=None, checkpoint_path=checkpoint_dir)
        else:
            model = Unet(**parameters)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.0025)
            model, opt, epoch, loss, dice = load_checkpoint(model, optimizer, scheduler=None, checkpoint_path=checkpoint_dir)

        if gpu:
            model.to('cuda')

        logger.info(f"Model {model_name} loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        raise

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/get-models')
def get_models():
    try:
        models = {k: v['name'] for k, v in MODEL_CONFIGS.items() if os.path.exists(v['checkpoint'])}
        current_model = {
            'key': CURRENT_MODEL['key'],
            'name': MODEL_CONFIGS[CURRENT_MODEL['key']]['name'] if CURRENT_MODEL['key'] else None
        }
        
        logger.info(f"Available models: {models}")
        logger.info(f"Current model: {current_model}")
        
        return jsonify({
            'models': models,
            'current_model': current_model
        })
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return jsonify({'error': str(e)}), 500
    

@app.route('/load-model', methods=['POST'])
def load_model_route():
    data = request.json
    model_key = data.get('model')
    
    if model_key not in MODEL_CONFIGS:
        return jsonify({'error': 'Invalid model selection'}), 400
    
    try:
        model_config = MODEL_CONFIGS[model_key]
        checkpoint_path = model_config['checkpoint']
        
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            return jsonify({'error': f'Checkpoint not found at {checkpoint_path}'}), 404


        model = load_model(
            parameters=model_config['params'], 
            model_name=model_key, 
            checkpoint_dir=checkpoint_path
        )
        
        CURRENT_MODEL['key'] = model_key
        CURRENT_MODEL['model'] = model
        
        return jsonify({
            'message': 'Model loaded successfully',
            'model_name': model_config['name'],
            'model_key': model_key
        })
        
    except Exception as e:
        logger.error(f"Model loading error: {e}")
        return jsonify({
            'error': f'Error loading model: {str(e)}'
        }), 500


@app.route('/view-uploads')
def view_uploads():
    return send_file(r"C:\Users\naufal\SKRIPSI_PROJECT\templates\view_uploaded_images.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    try:
        # Read the image using OpenCV
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img = np.array(Image.open(img_path))

        if img is None:
            raise Exception("Could not read the image")
        
        # Encode the image to send
        _, img_encoded = cv2.imencode('.png', img)
        
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    
    except Exception as e:
        return f"Error loading image: {str(e)}", 404

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files part'}), 400
    
    files = request.files.getlist('files[]')
    
    if not files:
        return jsonify({'error': 'No files selected'}), 400

    filenames = []
    errors = []
    print(len(files))
    for file in files:
        try:
            if file and file.filename:
                # Secure the filename
                original_filename = secure_filename(file.filename)
                png_filename = os.path.splitext(original_filename)[0] + '.png'
                
                # Open image with more lenient method
                try:
                    # Use PIL to open the image, which is more forgiving
                    img = Image.open(file)
                    
                    # Ensure image is in RGB mode
                    if img.mode not in ('RGB', 'RGBA'):
                        img = img.convert('RGB')
                    
                    # Save as PNG
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], png_filename)
                    img.save(filepath, 'PNG', optimize=True)
                    
                    filenames.append(png_filename)
                    
                except Exception as e:
                    errors.append(f"Error processing {file.filename}: {str(e)}")
                    
        except Exception as e:
            errors.append(f"Unexpected error with {file.filename}: {str(e)}")
    
    if not filenames:
        return jsonify({
            'error': 'No files were successfully processed',
            'details': errors
        }), 400
    
    return jsonify({
        'message': 'Files uploaded successfully', 
        'filenames': filenames,
        'errors': errors if errors else None
    }), 200

@app.route('/get-uploads')
def get_uploads():
    # Get list of files in upload directory
    upload_dir = r'C:\Users\naufal\SKRIPSI_PROJECT\uploads'
    files = [f for f in os.listdir(upload_dir) if os.path.isfile(os.path.join(upload_dir, f))]
    return jsonify({'filenames': files})


@app.route('/remove-upload', methods=['POST'])
def remove_upload():
    try:
        # Get the filename from the request
        data = request.json
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
        
        # Construct full path to the file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Check if file exists before trying to remove
        if os.path.exists(filepath):
            os.remove(filepath)
            
            # Optional: Also remove corresponding mask if it exists
            mask_filepath = os.path.join(app.config['RESULTS_FOLDER'], f'mask_{filename}')
            if os.path.exists(mask_filepath):
                os.remove(mask_filepath)
            
            return jsonify({'message': f'File {filename} removed successfully'}), 200
        else:
            return jsonify({'error': 'File not found'}), 404
    
    except Exception as e:
        logger.error(f"Error removing file {filename}: {e}")
        return jsonify({'error': str(e)}), 500



@app.route('/predict', methods=['POST'])
def predict():
    # Get list of files in upload folder


    if CURRENT_MODEL['model'] is None:
        return jsonify({'error': 'No model loaded'}), 400
        
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    files = [f for f in files]
    images = np.array(list(map(preprocessing, files)), dtype=np.float32)
    model =  CURRENT_MODEL['model']
    
    start = time()
    model.eval()
    with torch.no_grad():

        if len(images.shape) == 3:
            input_tensor = torch.tensor(images, dtype=torch.float).permute(2, 0, 1)
            input_tensor = input_tensor.unsqueeze(0)

        else:
            input_tensor = torch.tensor(images, dtype=torch.float).permute(0, 3, 1, 2)

        if GPU:
            input_tensor = input_tensor.cuda(non_blocking=True)
            torch.cuda.synchronize()


        mask_logits = model(input_tensor)
        mask_predictions = (mask_logits >= 0.5).float()
        mask_predictions = mask_predictions.view(-1, 512, 512).cpu().numpy()
        mask_predictions = (mask_predictions*255).astype(np.uint8)

        end = time()

    print(str(end-start), flush=True)
    # Save the prediction mask

    for i, mask in enumerate(mask_predictions):
        result_filename = f'mask_{files[i]}'
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        Image.fromarray(mask, mode='L').save(result_path)
    
    return jsonify({'message': 'Predictions completed', 
                   'results': mask_predictions.tolist()}), 200

@app.route('/view-results')
def view_results():
    return send_file(r"C:\Users\naufal\SKRIPSI_PROJECT\templates\view_result_mask.html")

@app.route('/results/<filename>')
def results_file(filename):
    try:
        img_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
        
        if not os.path.exists(img_path):
            return jsonify({'error': 'Image not found'}), 404
        
        # Read the image using PIL and convert to RGB for proper display
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Save to bytes
        img_io = io.BytesIO()
        img.save(img_io, 'PNG')
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/png')
        
    except Exception as e:
        print(f"Error serving image {filename}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get-results')
def show_results():
    try:
        files = [f for f in os.listdir(app.config['RESULTS_FOLDER']) 
                if os.path.isfile(os.path.join(app.config['RESULTS_FOLDER'], f))]
        files.sort()
        return jsonify({'filenames': files})
    except Exception as e:
        print(f"Error getting results: {str(e)}")  # For debugging
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=8765, debug=True)