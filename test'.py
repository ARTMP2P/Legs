import requests

# Установите путь к файлу архива, который вы хотите отправить
archive_file_path = '/input/archive.zip'

# URL-адрес API-сервера
api_url = 'http://localhost:80/images'

# объект запроса с архивом
files = {'image_file': open(archive_file_path, 'rb')}
data = {'mirr': 'mirr2', 'scale': '1'}

# POST-запрос к API
response = requests.post(api_url, files=files, data=data)

# Проверка статуса ответа
if response.status_code == 200:
    # Сохранить полученный архив на диск
    output_archive_path = '/path/to/save/output.zip'
    with open(output_archive_path, 'wb') as output_file:
        output_file.write(response.content)
    print('Обработанный архив сохранен в', output_archive_path)
else:
    print('Ошибка при отправке запроса:', response.text)
