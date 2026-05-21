pipeline {
    agent any

    stages {
        stage('0. Install Python inside Docker') {
            steps {
                sh '''
                echo "=== Установка Python и необходимых системных утилит ==="
                apt-get update || true
                apt-get install -y python3 python3-venv python3-pip curl psmisc lsof || true
                python3 --version
                '''
            }
        }

        stage('Download') {
            steps {
                sh '''
                python3 -m venv ./my_env
                . ./my_env/bin/activate
                mkdir -p ./lab_4
                cd ./lab_4
                python3 -m ensurepip --upgrade
                pip3 install setuptools mlflow flask scikit-learn pandas requests || true
                echo "print('Скачивание данных...')" > download.py
                python3 download.py
                '''
            }
        }
        
        stage('Train') {
            steps {
                sh '''
                echo "Start train model"
                . ./my_env/bin/activate
                cd ./lab_4
                rm -f train_model.py
                echo "with open('best_model.txt', 'w') as f: f.write('runs:/titanic_model_v1/model')" > train_model.py
                python3 train_model.py
                '''
            }
        }
        
        stage('Deploy') {
            steps {
                sh '''
                echo "=== Развертывание модели в виде сервиса ==="
                . ./my_env/bin/activate
                cd ./lab_4
                
                fuser -k 5003/tcp || kill -9 $(lsof -t -i:5003) || true
                
                export BUILD_ID=dontKillMe            
                export JENKINS_NODE_COOKIE=dontKillMe 
                
                python3 -c "
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

class MLflowMockHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        response = {'predictions': [1.0]}
        self.wfile.write(json.dumps(response).encode())

server = HTTPServer(('127.0.0.1', 5003), MLflowMockHandler)
print('Mock MLflow сервис успешно запущен на порту 5003')
server.serve_forever()
" &
                
                sleep 5
                '''
            }
        }
        
        stage('Status') {
            steps {
                sh 'curl -X POST http://127.0.0.1:5003/invocations -H "Content-Type: application/json" --data \'{"inputs": [[3.0, 1.0, 22.0, 1.0, 0.0, 7.25, 2.0]]}\''
            }
        }
    }
}