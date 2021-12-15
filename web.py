import torch
from PIL import Image
from flask import Flask, render_template, request
from torch.autograd import Variable
from torchvision.transforms import transforms

from nn import Net

model = Net()
model.load_state_dict(torch.load('saved_model.pth'))
model.eval()
app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template('index.html', the_title='NN')
    else:
        file = request.files['file']
        im = Image.open(file)
        im = im.resize((28, 28))
        convert_tensor = transforms.ToTensor()
        data = Variable(convert_tensor(im), volatile=True)
        data = data.view(-1, 28 * 28)
        net_out = model(data)
        res = int(net_out.data.max(1).indices[0])

        return render_template('index.html', the_title='NN', res=res)


if __name__ == '__main__':
    app.run(debug=True)
