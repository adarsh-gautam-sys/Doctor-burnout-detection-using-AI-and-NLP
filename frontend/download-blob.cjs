const https = require('https');
const fs = require('fs');

https.get("https://reactbits.dev/r/BlobCursor-JS-CSS.json", (res) => {
  let body = '';
  res.on('data', chunk => body += chunk);
  res.on('end', () => {
    const data = JSON.parse(body);
    data.files.forEach(f => {
      const fileName = f.name || f.path.split('/').pop();
      fs.writeFileSync("src/components/" + fileName, f.content);
      console.log("Saved", fileName);
    });
  });
});
