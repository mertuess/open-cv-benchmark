const img = document.getElementById('videoCanvas');
const histogramCanvas = document.getElementById('histogramCanvas');
const ctx = histogramCanvas.getContext('2d');
const ws = new WebSocket('ws://127.0.0.1:8000/video');

let currentPipeline = 'simple';
let pipelinesData = {};
let processorsData = {};
let currentParams = {};
let showHistogram = true;
let histogramType = 'rgb'; // 'rgb', 'gray', 'separate'

// Обработка сообщений от сервера
ws.onmessage = function(event) {
    try {
        const data = JSON.parse(event.data);
        if (data.type === 'initial_info') {
            // Сохраняем информацию
            pipelinesData = data.data.pipelines;
            processorsData = data.data.processors;
            
            // Создаем UI
            createPipelineSelector();
            createParametersUI(currentPipeline);
        } else if (data.type === 'pipeline_selected') {
            // Обновляем параметры при смене пайплайна
            currentPipeline = data.pipeline_id;
            createParametersUI(currentPipeline);
        } else {
            // Это изображение - обновляем img
            img.src = event.data;
            
            // Ждем загрузки изображения для отрисовки гистограммы
            if (showHistogram) {
                img.onload = function() {
                    drawHistogramFromImage();
                };
                // Если изображение уже загружено
                if (img.complete) {
                    drawHistogramFromImage();
                }
            } else {
                clearHistogram();
            }
        }
    } catch (e) {
        // Если не JSON - значит изображение
        img.src = event.data;
        if (showHistogram) {
            img.onload = function() {
                drawHistogramFromImage();
            };
            if (img.complete) {
                drawHistogramFromImage();
            }
        } else {
            clearHistogram();
        }
    }
};

// Функция для отрисовки гистограммы из изображения
function drawHistogramFromImage() {
    // Создаем временный canvas для извлечения данных пикселей
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = img.naturalWidth || img.width;
    tempCanvas.height = img.naturalHeight || img.height;
    const tempCtx = tempCanvas.getContext('2d');
    
    try {
        tempCtx.drawImage(img, 0, 0);
        const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
        const data = imageData.data;
        
        // В зависимости от типа гистограммы
        if (histogramType === 'rgb') {
            drawRGBHistogram(data);
        } else if (histogramType === 'gray') {
            drawGrayHistogram(data);
        } else if (histogramType === 'separate') {
            drawSeparateHistogram(data);
        }
    } catch (e) {
        console.error('Ошибка при создании гистограммы:', e);
        clearHistogram();
    }
}

// Отрисовка RGB гистограммы (все каналы вместе)
function drawRGBHistogram(data) {
    const width = histogramCanvas.width;
    const height = histogramCanvas.height;
    const histSize = 256;
    
    // Инициализируем гистограммы
    const histR = new Array(histSize).fill(0);
    const histG = new Array(histSize).fill(0);
    const histB = new Array(histSize).fill(0);
    
    // Заполняем гистограммы
    for (let i = 0; i < data.length; i += 4) {
        histR[data[i + 2]]++;
        histG[data[i + 1]]++;
        histB[data[i]]++;
    }
    
    // Находим максимум для нормализации
    let max = 0;
    for (let i = 0; i < histSize; i++) {
        max = Math.max(max, histR[i], histG[i], histB[i]);
    }
    max = Math.max(max, 1);
    
    // Очищаем canvas
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, width, height);
    
    // Рисуем гистограммы
    const barWidth = width / histSize;
    const scale = (height - 20) / max;
    const offsetY = 10;
    
    for (let i = 0; i < histSize; i++) {
        const x = i * barWidth;
        
        // Red
        ctx.fillStyle = 'rgba(255, 0, 0, 0.5)';
        ctx.fillRect(x, height - (histR[i] * scale + offsetY), barWidth + 1, histR[i] * scale);
        
        // Green
        ctx.fillStyle = 'rgba(0, 255, 0, 0.5)';
        ctx.fillRect(x, height - (histG[i] * scale + offsetY), barWidth + 1, histG[i] * scale);
        
        // Blue
        ctx.fillStyle = 'rgba(0, 0, 255, 0.5)';
        ctx.fillRect(x, height - (histB[i] * scale + offsetY), barWidth + 1, histB[i] * scale);
    }
    
    // Рисуем легенду
    drawLegend(['R', 'G', 'B'], ['#ff0000', '#00ff00', '#0000ff']);
}

// Отрисовка гистограммы в оттенках серого
function drawGrayHistogram(data) {
    const width = histogramCanvas.width;
    const height = histogramCanvas.height;
    const histSize = 256;
    
    const hist = new Array(histSize).fill(0);
    
    // Конвертируем в оттенки серого и заполняем гистограмму
    for (let i = 0; i < data.length; i += 4) {
        const gray = Math.round(0.299 * data[i + 2] + 0.587 * data[i + 1] + 0.114 * data[i]);
        hist[gray]++;
    }
    
    // Находим максимум для нормализации
    let max = Math.max(...hist, 1);
    
    // Очищаем canvas
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, width, height);
    
    // Рисуем гистограмму
    const barWidth = width / histSize;
    const scale = (height - 20) / max;
    const offsetY = 10;
    
    for (let i = 0; i < histSize; i++) {
        const x = i * barWidth;
        const barHeight = hist[i] * scale;
        const gray = Math.round(i / histSize * 255);
        ctx.fillStyle = `rgb(${gray}, ${gray}, ${gray})`;
        ctx.fillRect(x, height - (barHeight + offsetY), barWidth + 1, barHeight);
    }
    
    // Рисуем легенду
    drawLegend(['Gray'], ['#ffffff']);
}

// Отрисовка раздельной гистограммы (3 канала отдельно)
function drawSeparateHistogram(data) {
    const width = histogramCanvas.width;
    const height = histogramCanvas.height;
    const histSize = 256;
    const channelHeight = (height - 20) / 3;
    
    // Инициализируем гистограммы
    const histR = new Array(histSize).fill(0);
    const histG = new Array(histSize).fill(0);
    const histB = new Array(histSize).fill(0);
    
    // Заполняем гистограммы
    for (let i = 0; i < data.length; i += 4) {
        histR[data[i + 2]]++;
        histG[data[i + 1]]++;
        histB[data[i]]++;
    }
    
    // Находим максимумы для нормализации
    let maxR = Math.max(...histR, 1);
    let maxG = Math.max(...histG, 1);
    let maxB = Math.max(...histB, 1);
    
    // Очищаем canvas
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, width, height);
    
    const barWidth = width / histSize;
    const offsetY = 5;
    
    // Red канал (верх)
    const scaleR = (channelHeight - offsetY) / maxR;
    for (let i = 0; i < histSize; i++) {
        const x = i * barWidth;
        ctx.fillStyle = 'rgba(255, 0, 0, 0.7)';
        ctx.fillRect(x, channelHeight - (histR[i] * scaleR + offsetY), barWidth + 1, histR[i] * scaleR);
    }
    ctx.fillStyle = '#ff0000';
    ctx.font = '10px Arial';
    ctx.fillText('R', 5, 12);
    
    // Green канал (средний)
    const scaleG = (channelHeight - offsetY) / maxG;
    for (let i = 0; i < histSize; i++) {
        const x = i * barWidth;
        ctx.fillStyle = 'rgba(0, 255, 0, 0.7)';
        ctx.fillRect(x, channelHeight + channelHeight - (histG[i] * scaleG + offsetY), barWidth + 1, histG[i] * scaleG);
    }
    ctx.fillStyle = '#00ff00';
    ctx.font = '10px Arial';
    ctx.fillText('G', 5, channelHeight + 12);
    
    // Blue канал (нижний)
    const scaleB = (channelHeight - offsetY) / maxB;
    for (let i = 0; i < histSize; i++) {
        const x = i * barWidth;
        ctx.fillStyle = 'rgba(0, 0, 255, 0.7)';
        ctx.fillRect(x, height - (histB[i] * scaleB + offsetY), barWidth + 1, histB[i] * scaleB);
    }
    ctx.fillStyle = '#0000ff';
    ctx.font = '10px Arial';
    ctx.fillText('B', 5, height - 5);
}

// Отрисовка легенды
function drawLegend(labels, colors) {
    const legendY = 5;
    let x = 10;
    
    ctx.font = '10px Arial';
    labels.forEach((label, index) => {
        ctx.fillStyle = colors[index];
        ctx.fillRect(x, legendY, 10, 10);
        ctx.fillText(label, x + 14, legendY + 10);
        x += 50;
    });
}

// Очистка гистограммы
function clearHistogram() {
    ctx.clearRect(0, 0, histogramCanvas.width, histogramCanvas.height);
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, histogramCanvas.width, histogramCanvas.height);
    ctx.fillStyle = '#555';
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Гистограмма отключена', histogramCanvas.width / 2, histogramCanvas.height / 2 + 5);
    ctx.textAlign = 'left';
}

// Создание селектора пайплайнов
function createPipelineSelector() {
    const container = document.querySelector('.option-panel');
    
    // Удаляем старый селектор если есть
    const oldSelector = container.querySelector('.pipeline-selector');
    if (oldSelector) oldSelector.remove();
    
    const wrapper = document.createElement('div');
    wrapper.className = 'pipeline-selector';
    
    const label = document.createElement('label');
    label.textContent = 'Выберите пайплайн:';
    wrapper.appendChild(label);
    
    const select = document.createElement('select');
    select.id = 'pipelineSelect';
    
    Object.values(pipelinesData).forEach(pipeline => {
        const option = document.createElement('option');
        option.value = pipeline.id;
        option.textContent = pipeline.name;
        if (pipeline.id === currentPipeline) {
            option.selected = true;
        }
        select.appendChild(option);
    });
    
    select.addEventListener('change', () => {
        const id = select.value;
        if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
                type: 'select_pipeline',
                pipeline_id: id
            }));
        }
        currentPipeline = id;
    });
    
    wrapper.appendChild(select);
    container.prepend(wrapper);
}

// Создание UI для параметров текущего пайплайна
function createParametersUI(pipelineId) {
    const pipeline = pipelinesData.find(p => p.id === pipelineId);
    if (!pipeline) return;
    
    const container = document.querySelector('.option-panel');
    const paramsContainer = container.querySelector('.params-container');
    
    if (paramsContainer) paramsContainer.remove();
    
    const newContainer = document.createElement('div');
    newContainer.className = 'params-container';
    
    const desc = document.createElement('p');
    desc.className = 'pipeline-description';
    desc.textContent = pipeline.description;
    newContainer.appendChild(desc);
    
    pipeline.processors.forEach(processorId => {
        const processor = processorsData.find(p => p.id === processorId);
        if (!processor) return;
        
        const processorBlock = document.createElement('div');
        processorBlock.className = 'processor-block';
        
        const title = document.createElement('h3');
        title.textContent = processor.name;
        processorBlock.appendChild(title);
        
        processor.params.forEach(paramDef => {
            const wrapper = document.createElement('div');
            wrapper.className = 'param-group';
            
            const label = document.createElement('label');
            label.htmlFor = `${processorId}_${paramDef.name}`;
            label.textContent = paramDef.label;
            wrapper.appendChild(label);
            
            const currentValue = getParamValue(pipelineId, processorId, paramDef);
            
            if (paramDef.type === 'select') {
                const select = document.createElement('select');
                select.id = `${processorId}_${paramDef.name}`;
                select.name = paramDef.name;
                
                paramDef.options.forEach(option => {
                    const opt = document.createElement('option');
                    opt.value = option.value;
                    opt.textContent = option.label;
                    if (option.value === currentValue) {
                        opt.selected = true;
                    }
                    select.appendChild(opt);
                });
                
                select.addEventListener('change', () => {
                    const value = paramDef.options[select.selectedIndex].value;
                    updateParam(pipelineId, processorId, paramDef.name, value);
                });
                
                wrapper.appendChild(select);
            } else if (paramDef.type === 'range') {
                const input = document.createElement('input');
                input.type = 'range';
                input.id = `${processorId}_${paramDef.name}`;
                input.name = paramDef.name;
                input.min = paramDef.min;
                input.max = paramDef.max;
                input.step = paramDef.step || 1;
                input.value = currentValue || paramDef.default;
                
                const valueDisplay = document.createElement('span');
                valueDisplay.className = 'value-display';
                valueDisplay.textContent = input.value;
                
                input.addEventListener('input', () => {
                    const value = input.value;
                    valueDisplay.textContent = value;
                    updateParam(pipelineId, processorId, paramDef.name, value);
                });
                
                wrapper.appendChild(input);
                wrapper.appendChild(valueDisplay);
            } else if (paramDef.type === 'checkbox') {
                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.id = `${processorId}_${paramDef.name}`;
                checkbox.name = paramDef.name;
                checkbox.checked = currentValue !== undefined ? currentValue : paramDef.default;
                
                checkbox.addEventListener('change', () => {
                    updateParam(pipelineId, processorId, paramDef.name, checkbox.checked);
                });
                
                wrapper.appendChild(checkbox);
            }
            
            processorBlock.appendChild(wrapper);
        });
        
        newContainer.appendChild(processorBlock);
    });
    
    container.appendChild(newContainer);
}

function getParamValue(pipelineId, processorId, paramDef) {
    if (!currentParams[pipelineId]) {
        currentParams[pipelineId] = {};
    }
    if (!currentParams[pipelineId][processorId]) {
        currentParams[pipelineId][processorId] = {};
    }
    if (currentParams[pipelineId][processorId][paramDef.name] === undefined) {
        return paramDef.default;
    }
    return currentParams[pipelineId][processorId][paramDef.name];
}

function updateParam(pipelineId, processorId, paramName, value) {
    if (!currentParams[pipelineId]) {
        currentParams[pipelineId] = {};
    }
    if (!currentParams[pipelineId][processorId]) {
        currentParams[pipelineId][processorId] = {};
    }
    
    currentParams[pipelineId][processorId][paramName] = value;
    
    if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            type: 'update_params',
            pipeline_id: pipelineId,
            processor_id: processorId,
            params: {
                [paramName]: value
            }
        }));
    }
}

// Настройка элементов управления гистограммой
document.addEventListener('DOMContentLoaded', function() {
    const showCheckbox = document.getElementById('showHistogram');
    const typeRadios = document.querySelectorAll('input[name="histogramType"]');
    const histogramContainer = document.querySelector('.histogram-container');
    
    if (showCheckbox) {
        showCheckbox.addEventListener('change', function() {
            showHistogram = this.checked;
            if (showHistogram) {
                histogramContainer.style.display = 'block';
                drawHistogramFromImage();
            } else {
                histogramContainer.style.display = 'none';
            }
        });
    }
    
    typeRadios.forEach(radio => {
        radio.addEventListener('change', function() {
            if (this.checked) {
                histogramType = this.value;
                if (showHistogram) {
                    drawHistogramFromImage();
                }
            }
        });
    });
    
    // Начальное состояние
    const initialShow = document.querySelector('#showHistogram');
    if (initialShow) {
        showHistogram = initialShow.checked;
        if (!showHistogram) {
            histogramContainer.style.display = 'none';
        }
    }
});

ws.onerror = function(error) {
    console.error('Ошибка WebSocket: ', error);
};

ws.onopen = function() {
    console.log('WebSocket соединение установлено');
};
