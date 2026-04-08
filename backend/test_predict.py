"""Quick test for the V5 /predict endpoint."""
import http.client, json, os, glob

# Find a real prescription image
imgs = glob.glob(r'..\CliniCare Dataset\real\*.jpg')
if not imgs:
    imgs = glob.glob(r'..\CliniCare Dataset\real\*.png')
if not imgs:
    print("No test images found!")
    exit(1)

test_img = imgs[0]
print(f"Testing with: {os.path.basename(test_img)}")

# Build multipart/form-data body
boundary = '----TestBoundary12345'
filename = os.path.basename(test_img)
with open(test_img, 'rb') as f:
    file_data = f.read()

body = b''
body += f'--{boundary}\r\n'.encode()
body += f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode()
body += b'Content-Type: image/jpeg\r\n\r\n'
body += file_data
body += f'\r\n--{boundary}--\r\n'.encode()

headers = {
    'Content-Type': f'multipart/form-data; boundary={boundary}',
}

conn = http.client.HTTPConnection('127.0.0.1', 8000, timeout=120)
conn.request('POST', '/predict', body, headers)
res = conn.getresponse()
data = json.loads(res.read())

print(f"\nStatus: {res.status}")
print(json.dumps(data, indent=2))
