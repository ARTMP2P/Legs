import requests

# Установите путь к файлу архива, который вы хотите отправить
archive_file_path = './in/archive.zip'

# URL-адрес API-сервера
api_url = 'http://192.168.8.178:8080/images'

# Объект запроса с архивом
files = {'image_file': open(archive_file_path, 'rb')}
data = {'mirr': 'mirr2'}

# POST-запрос к API
response = requests.post(api_url, files=files, data=data)

# Проверка статуса ответа
if response.status_code == 200:
    # Сохранить полученный архив на диск
    output_archive_path = './output/output.zip'
    with open(output_archive_path, 'wb') as output_file:
        output_file.write(response.content)
    print('Обработанный архив сохранен в', output_archive_path)
else:
    print('Ошибка при отправке запроса:', response.text)

# Запись характеристик ответа и содержимого в файл
result_file_path = './output/result.txt'
with open(result_file_path, 'w') as result_file:
    result_file.write(f'Status Code: {response.status_code}\n')
    result_file.write('Response Headers:\n')
    for header, value in response.headers.items():
        result_file.write(f'{header}: {value}\n')
    result_file.write('\nResponse Body:\n')
    result_file.write(response.text)
print('Характеристики ответа и содержимое записаны в', result_file_path)
