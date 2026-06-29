import asyncio
import cv2
import base64
import json
from fastapi import FastAPI, WebSocket
import uvicorn
from processors import processor_manager
from pipelines.predefined_pipelines import ALL_PIPELINES

app = FastAPI()

current_pipeline_id = "simple"
pipeline_params = {}

pipelines = {p.id: p for p in ALL_PIPELINES}

for pipeline in ALL_PIPELINES:
    pipeline_params[pipeline.id] = pipeline.get_all_params(
        processor_manager._processors)


@app.websocket("/video")
async def video_stream(websocket: WebSocket):
    await websocket.accept()

    cap = None
    for camera_id in range(3):
        cap = cv2.VideoCapture(camera_id)
        if cap.isOpened():
            print(f"Камера {camera_id} открыта")
            break
        else:
            cap.release()
            cap = None

    if cap is None:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "Не удалось открыть камеру"
        }))
        await websocket.close()
        return

    await send_initial_info(websocket)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Не удалось получить кадр")
                break

            try:
                pipeline = pipelines.get(current_pipeline_id)
                if pipeline:
                    params = pipeline_params.get(current_pipeline_id, {})
                    frame = pipeline.process(
                        frame, processor_manager._processors, params)
            except Exception as e:
                print(f"Ошибка при обработке кадра: {e}")

            try:
                _, buffer = cv2.imencode(
                    '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                await websocket.send_text(f"data:image/jpeg;base64,{jpg_as_text}")
            except Exception as e:
                print(f"Ошибка при отправке кадра: {e}")
                break

            try:
                message = await asyncio.wait_for(websocket.receive_text(),
                                                 timeout=0.001)
                data = json.loads(message)
                await handle_message(websocket, data)
            except asyncio.TimeoutError:
                pass
            except json.JSONDecodeError as e:
                print(f"Ошибка парсинга JSON: {e}")
            except Exception as e:
                print(f"Ошибка при получении сообщения: {e}")

            await asyncio.sleep(0.03)

    except Exception as e:
        print(f"Ошибка в основном цикле: {e}")
    finally:
        if cap:
            cap.release()
        print("Камера освобождена")


async def send_initial_info(websocket: WebSocket):
    """Отправляет информацию о пайплайнах и процессорах"""
    try:
        info = {
            "type": "initial_info",
            "data": {
                "pipelines": [
                    {
                        "id": p.id,
                        "name": p.name,
                        "description": p.description,
                        "processors": p.get_processor_chain()
                    }
                    for p in ALL_PIPELINES
                ],
                "processors": [
                    {
                        "id": p.id,
                        "name": p.name,
                        "category": p.category,
                        "params": p.param_definitions
                    }
                    for p in processor_manager.get_all_processors()
                ]
            }
        }
        await websocket.send_text(json.dumps(info))
    except Exception as e:
        print(f"Ошибка при отправке initial_info: {e}")


async def handle_message(websocket: WebSocket, data: dict):
    """Обрабатывает сообщения от клиента"""
    global current_pipeline_id, pipeline_params

    msg_type = data.get("type")

    if msg_type == "select_pipeline":
        pipeline_id = data.get("pipeline_id")
        if pipeline_id in pipelines:
            current_pipeline_id = pipeline_id
            print(f"Выбран пайплайн: {pipeline_id}")

            # Отправляем подтверждение
            await websocket.send_text(json.dumps({
                "type": "pipeline_selected",
                "pipeline_id": pipeline_id
            }))

    elif msg_type == "update_params":
        pipeline_id = data.get("pipeline_id", current_pipeline_id)
        processor_id = data.get("processor_id")
        params = data.get("params", {})

        if pipeline_id in pipeline_params:
            if processor_id in pipeline_params[pipeline_id]:
                for key, value in params.items():
                    if key in pipeline_params[pipeline_id][processor_id]:
                        current_value = pipeline_params[pipeline_id][processor_id][key]
                        if isinstance(current_value, bool):
                            pipeline_params[pipeline_id][processor_id][key] = bool(
                                value)
                        elif isinstance(current_value, float):
                            pipeline_params[pipeline_id][processor_id][key] = float(
                                value)
                        elif isinstance(current_value, int):
                            pipeline_params[pipeline_id][processor_id][key] = int(
                                value)
                        else:
                            pipeline_params[pipeline_id][processor_id][key] = value

                print(f"Обновлены параметры: {pipeline_id} -> {processor_id}")

                await websocket.send_text(json.dumps({
                    "type": "params_updated",
                    "pipeline_id": pipeline_id,
                    "processor_id": processor_id
                }))

    elif msg_type == "ping":
        await websocket.send_text(json.dumps({"type": "pong"}))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
