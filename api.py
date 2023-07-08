import tensorflow as tf
import time
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    # tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_virtual_device_configuration(
        gpu,
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
    )


import time, os, werkzeug, zipfile, cv2, shutil
from flask import Flask, render_template, make_response, request, Blueprint, send_file
from flask_restx import Api, Resource, fields, reqparse
from werkzeug.utils import secure_filename
from functions.function import get_img_for_predict, predict_img, save_gen_img, get_image_size

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# допустимые разсширения имен файлов с изображениями
ALLOWED_EXTENSIONS = {'zip', 'rar', '7z', 'tar', 'gz'}

# имя каталога для загрузки изображений
UPLOAD_FOLDER = 'inputs/'
RESULT_FOLDER = 'result/'


def allowed_file(file_name):
    """
    Функция проверки расширения файла
    """
    return '.' in file_name and file_name.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def add_affix(filename, affix='_CENTR'):
    name, extension = os.path.splitext(filename)
    return name + affix + '.png'


def zip_folder(name):
    print(name)
    zip_name = name + '.zip'
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
        for folder_name, subfolders, filenames in os.walk(name):
            for filename in filenames:
                file_path = os.path.join(folder_name, filename)
                zip_ref.write(file_path, arcname=os.path.relpath(file_path, name))

    zip_ref.close()


# создаем WSGI приложение
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# создаем API-сервер
api = Api(app)

# создаем парсер API-запросов
parser = reqparse.RequestParser()
parser.add_argument('image_file', type=werkzeug.datastructures.FileStorage, help='Binary Image in png format (zip, rar, 7z, tar, gz)', location='files', required=True)
parser.add_argument('mirror', type=werkzeug.datastructures.Accept, required=True)


@api.route('/images', methods=['GET', 'POST'])
@api.produces(['/application'])
class Images(Resource):
    # если POST-запрос
    @api.expect(parser)
    def post(self):

        global width, height
        try:
            # определяем текущее время
            start_time = time.time()

            # проверка наличия файла в запросе
            if 'image_file' not in request.files:
                raise ValueError('No input file')


            # получаем файл из запроса
            f = request.files['image_file']

            req_mirr = request.form.get('mirr');
            # print(f"mirr: {req_mirr}")
            # проверка на наличие имени у файла
            if f.filename == '':
                raise ValueError('Empty file name')

            # проверка на допустимое расширение файла (png, jpg, jpeg, tga, dds)
            if not allowed_file(f.filename):
                raise ValueError('Upload an image in one of the formats (zip, rar, 7z, tar, gz)')

            # имя файла
            image_file_name = secure_filename(f.filename)
            # print(f"image_file_name: {image_file_name}")
            # задаем полный путь к файлу
            image_file_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file_name)
            # сохраняем файл
            f.save(image_file_path)
            # print(f"image_file_path: {image_file_path}")
            try:
                unzipped = os.path.join(app.config['UPLOAD_FOLDER'], image_file_name.split('.')[0] + time.time().__str__())
                img_path_save = os.path.join(app.config['RESULT_FOLDER'], image_file_name.split('.')[0])
                if not os.path.exists(img_path_save):
                    os.mkdir(img_path_save)

                # unzipping files
                with zipfile.ZipFile(image_file_path, 'r') as zip_ref:
                    zip_ref.extractall(unzipped)

                time.sleep(3)

                # сортируем список файлов в директории
                files = os.listdir(unzipped)
                files.sort()

                # сохраняем оригинальные имена файлов, создаем список новых имен файлов
                #new_file_names_mirr2 = ["0", "035", "090", "145", "180", "215", "270", "325"]
                new_file_names_mirr2 = ["a", "b", "c", "d", "e", "f", "g", "h"]
                #new_file_names_mirr1 = ["180", "145", "090", "035", "0", "325", "270", "215"]
                new_file_names_mirr1 = ["i", "j", "k", "l", "m", "n", "o", "p"]

                new_file_names = new_file_names_mirr1 if req_mirr == 'mirr1' else new_file_names_mirr2

                # original_file_names_with_affix = list(map(add_affix, original_file_names))
            except Exception as e:
                print(f"Ошибка при формировании имен файлов: {e}")
            # переименовываем файлы
            file_counter = 0
            for filename in files:
                if filename.endswith('.png'):
                    try:
                        width, height = get_image_size(os.path.join(unzipped, filename))
                        os.rename(os.path.join(unzipped, filename),
                                  os.path.join(unzipped, new_file_names[file_counter] + ".png"))

                        file_counter += 1
                    except Exception as e:
                        print(f'Ошибка при переименовании файла: {filename}\n{e}')
            print(type(get_img_for_predict(unzipped)), get_img_for_predict(unzipped).shape)
            imgs = predict_img(get_img_for_predict(unzipped))
            print(f"predict_img ==================OK")
            save_gen_img(imgs, img_path_save, width, height)
            print(f"save_gen_img =================OK")
            # переименовываем файлы обратно
            for i in range(8):
                os.rename(os.path.join(img_path_save, f'{str(i)}.png'), os.path.join(img_path_save, new_file_names_mirr2[i] + '.png'))

            # конвертируем в gray
            for filename in os.listdir(img_path_save):
                if filename.endswith('.png'):
                    img = cv2.imread(os.path.join(img_path_save, filename))
                    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(os.path.join(img_path_save, filename), gray_image)
            print(f"Convert gray =================OK")
            zip_folder(img_path_save)

            os.remove(image_file_path)
            shutil.rmtree(unzipped)
            shutil.rmtree(img_path_save)

            zip_file = img_path_save + '.zip';

            response = send_file(zip_file, download_name=os.path.basename(zip_file), as_attachment=True, mimetype='application/zip')
            os.remove(zip_file)
            return response

        except ValueError as err:
            dict_response = {
                'error': err.args[0],
                'filename': f.filename,
                'time': (time.time() - start_time)
            }
            return dict_response

        except:
            dict_response = {
                'error': 'Unknown error',
                'time': (time.time() - start_time)
            }
            return dict_response


# запускаем сервер на порту 8008 (или на любом другом свободном порту)
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8080)


