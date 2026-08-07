git clone https://github.com/sutirtharana2005-netizen/project.git

py -m venv venv

venv/Scripts/activate

pip install torch torchvision pillow

pip install onnx onnxscript onnxruntime 


python -c "import onnx, onnxruntime as ort; m=onnx.load('artifacts/model.onnx'); print('ONNX opset:', [o.version for o in m.opset_import]); ort.InferenceSession('artifacts/model.onnx'); print('ONNX loaded in onnxruntime OK')"   

cd backend

pip install -r requirements.txt

uvicorn main:app --host 0.0.0.0 --port 8000
