from glob_inc.utils import *
import paho.mqtt.publish as publish
import paho.mqtt.subscribe as subscribe
import json
#from model_api.src.ml_api import start_training_task
from mnist import start_training_task_noniid
#broker_name = "100.95.25.52"
broker_name = "192.168.101.210"
# global start_line
start_line = 0
start_benign = 0

# percent_main_dga = 0.8
# 1 round
# total_data_dgas = 1980
num_line = 50
num_file = 1


# chi tinh main dga nhung so luong benign van = dgas
# len_dga_types = 1

def do_evaluate_connection(client):
    print_log("doing ping")
    client_id = client._client_id.decode("utf-8")
    result = ping_host(broker_name)
    result["client_id"] = client_id
    result["task"] = "EVA_CONN"
    client.publish(topic="dynamicFL/res/" + client_id, payload=json.dumps(result))
    print_log(f"publish to topic dynamicFL/res/{client_id}")
    return result


def do_evaluate_data():
    pass

def do_train(client):
    global start_line
    global start_benign


    print_log(f"start training")
    client_id = client._client_id.decode("utf-8")
    result, protos = start_training_task_noniid()

    start_line = start_line + num_line
    start_benign = start_benign + 10*num_line

    # Convert tensors to numpy arrays
    result_np = {key: value.cpu().numpy().tolist() for key, value in result.items()}
    payload = {
        "task": "TRAIN",
        "weight": result_np,
        "protos": protos
    }
    client.publish(topic="dynamicFL/res/" + client_id, payload=json.dumps(payload))
    print_log(f"end training")

def do_test():
    pass


def do_update_model():
    pass


def do_stop_client(client):
    print_log("stop client")
    client.loop_stop()


def handle_task(msg, client):
    task_name = msg.payload.decode("utf-8")
    if task_name == "EVA_CONN":
        do_evaluate_connection(client)
    elif task_name == "EVA_DATA":
        do_evaluate_data(client)
    elif task_name == "TRAIN":
        do_train(client)
    elif task_name == "TEST":
        do_test(client)
    elif task_name == "UPDATE":
        do_update_model(client)
    elif task_name == "REJECTED":
        do_add_errors(client)
    elif task_name == "STOP":
        do_stop_client(client)
    else:
        print_log(f"Command {task_name} is not supported")


def join_dFL_topic(client):
    client_id = client._client_id.decode("utf-8")
    client.publish(topic="dynamicFL/join", payload=client_id)
    print_log(f"{client_id} joined dynamicFL/join of {broker_name}")


def do_add_errors(client_id):
    publish.single(topic="dynamicFL/errors", payload=client_id, hostname=broker_name, client_id=client_id)


def wait_for_model(client_id):
    msg = subscribe.simple("dynamicFL/model", hostname=broker_name)
    fo = open("mymodel.pt", "wb")
    fo.write(msg.payload)
    fo.close()
    print_log(f"{client_id} write model to mymodel.pt")


def handle_cmd(client, userdata, msg):
    print_log("wait for cmd")
    client_id = client._client_id.decode("utf-8")
    handle_task(msg, client)
    print_log(f"{client_id} finished task {msg.payload.decode()}")


def handle_model(client, userdata, msg):
    print_log("receive model")
    f = open("newmode.pt", "wb")
    f.write(msg.payload)
    f.close()
    print_log("done write model")
    client_id = client._client_id.decode("utf-8")
    result = {
        "client_id": client_id,
        "task": "WRITE_MODEL"
    }
    client.publish(topic="dynamicFL/res/" + client_id, payload=json.dumps(result))
