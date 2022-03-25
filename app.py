import io
from flask import Flask, request, send_file
from clarendon import clarendon
from xpro import xpro
from cartoon import cartoon
from moon import moon
from kelvin import kelvin
from colorpop import colorpop

app = Flask(__name__)

uploads = 'uploads'
app.config['UPLOAD_FOLDER'] = uploads

# @app.route('/')
# def hello():
#     return '<h1>Hello, World!</h1>'


@app.route('/<filter>', methods=['POST'])
def apply_filter(filter):
    try:
        imageFile = request.files['file'].read()
    except Exception as e:
        print(e)
        return e
    else:
        if filter == 'clarendon':
            image_buffer= clarendon(imageFile)
            return send_file(io.BytesIO(image_buffer),
                                    attachment_filename='clarendon.jpg',
                                    mimetype='image/jpeg')
        if filter == 'xpro':
            image_buffer= xpro(imageFile)
            return send_file(io.BytesIO(image_buffer),
                                    attachment_filename='xpro.jpg',
                                    mimetype='image/jpeg')
        
        if filter == 'cartoon':
            image_buffer= cartoon(imageFile)
            return send_file(io.BytesIO(image_buffer),
                                    attachment_filename='cartoon.jpg',
                                    mimetype='image/jpeg')

        if filter == 'moon':
            image_buffer= moon(imageFile)
            return send_file(io.BytesIO(image_buffer),
                                    attachment_filename='moon.jpg',
                                    mimetype='image/jpeg')

        if filter == 'kelvin':
            image_buffer= kelvin(imageFile)
            return send_file(io.BytesIO(image_buffer),
                                    attachment_filename='kelvin.jpg',
                                    mimetype='image/jpeg')
        

        if filter== 'colorpop':
            image_buffer= colorpop(imageFile)
            return image_buffer
            # return send_file(io.BytesIO(image_buffer),
            #                         attachment_filename='colorpop.jpg',
            #                         mimetype='image/jpeg')

        return 'hi'
 





if __name__ == '__main__':
    app.run(host="localhost", port=3000, debug=True)