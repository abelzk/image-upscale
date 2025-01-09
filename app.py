import os
import subprocess
import shutil
import zipfile

# Constants
CODEFORMER_FIDELITY = 0.7
BACKGROUND_ENHANCE = True
FACE_UPSAMPLE = False

# Set up the environment
def setup_environment():
    subprocess.run(['pip', 'install', '-r', 'requirements.txt'], check=True)
    subprocess.run(['python', 'basicsr/setup.py', 'develop'], check=True)

# Download pre-trained models
def download_pretrained_models(model_name):
    subprocess.run(['python', 'scripts/download_pretrained_models.py', model_name], check=True)

# Extract images from inputs.zip
def extract_images(zip_file_path, extract_to):
    if os.path.exists(extract_to):
        shutil.rmtree(extract_to)
    os.mkdir(extract_to)
    
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f'Extracted images to {extract_to}')

# Run the upscale process
def upscale_images(input_path):
    if BACKGROUND_ENHANCE:
        if FACE_UPSAMPLE:
            subprocess.run([
                'python', 'inference_codeformer.py', 
                '-w', str(CODEFORMER_FIDELITY), 
                '--input_path', input_path, 
                '--bg_upsampler', 'realesrgan', 
                '--face_upsample'
            ], check=True)
        else:
            subprocess.run([
                'python', 'inference_codeformer.py', 
                '-w', str(CODEFORMER_FIDELITY), 
                '--input_path', input_path, 
                '--bg_upsampler', 'realesrgan'
            ], check=True)
    else:
        subprocess.run([
            'python', 'inference_codeformer.py', 
            '-w', str(CODEFORMER_FIDELITY), 
            '--input_path', input_path
        ], check=True)

    # Compress results
    result_dir = f'results/user_upload_{CODEFORMER_FIDELITY}/final_results'
    result_zip = 'results.zip'
    shutil.make_archive('results', 'zip', result_dir)
    print('Results compressed into', result_zip)

# Main execution
if __name__ == '__main__':
    folder_name = 'CodeFormer'
    zip_file_path = 'inputs.zip'
    extract_folder = 'inputs/user_upload'

    if os.path.exists(folder_name):
        # Change working directory to the pre-downloaded repository
        os.chdir(folder_name)
        
        setup_environment()
        
        # Download required models
        download_pretrained_models('facelib')
        download_pretrained_models('CodeFormer')
        
        # Check for and extract images from inputs.zip
        if os.path.exists(f'../{zip_file_path}'):
            extract_images(f'../{zip_file_path}', extract_folder)
            # Run the upscale process
            upscale_images(extract_folder)
            
            print('Download results.zip to get the processed images.')
        else:
            print(f'Zip file {zip_file_path} not found. Please upload it and run the script again.')
    else:
        print(f'Repository folder {folder_name} not found. Please ensure the CodeFormer repository is downloaded in the project directory.')