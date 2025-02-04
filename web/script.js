// web/script.js

document.addEventListener('DOMContentLoaded', () => {
    // Fetch audio sources, device list, supported languages, and current settings on load
    eel.get_audio_sources();
    eel.get_device_list();
    eel.get_supported_languages();
    eel.get_current_settings();

    // Elements
    const audioSourceSelect = document.getElementById('audioSource');
    const deviceSelector = document.getElementById('deviceSelector');
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const rawTextArea = document.getElementById('rawText');
    const translatedTextArea = document.getElementById('translatedText');
    const statusBar = document.getElementById('statusBar');
    const sourceLanguageSelect = $('#sourceLanguage');
    const targetLanguageSelect = $('#targetLanguage');

    // Initialize Selectize for language selectors after ensuring Selectize is loaded
    if (sourceLanguageSelect.length && targetLanguageSelect.length) {
        sourceLanguageSelect.selectize({
            create: false,
            sortField: 'text',
            placeholder: 'Select source language...',
            maxOptions: 1000,
            plugins: ['remove_button'],
            dropdownParent: 'body' // Ensure dropdown appears correctly
        });

        targetLanguageSelect.selectize({
            create: false,
            sortField: 'text',
            placeholder: 'Select target language...',
            maxOptions: 1000,
            plugins: ['remove_button'],
            dropdownParent: 'body' // Ensure dropdown appears correctly
        });
    }

    // Function to display status messages
    function showStatus(message, type = 'info') {
        statusBar.className = ''; // Reset classes
        statusBar.classList.add('alert', `alert-${type}`, 'mt-3');
        statusBar.textContent = message;
        statusBar.classList.remove('d-none'); // Show the status bar
        // Automatically hide after 5 seconds
        setTimeout(() => {
            statusBar.classList.add('d-none');
        }, 5000);
    }

    // Start Button Click
    startBtn.addEventListener('click', () => {
        const selectedAudioIndex = audioSourceSelect.value;
        const selectedDevice = deviceSelector.value;
        const sourceLanguage = sourceLanguageSelect[0].selectize.getValue();
        const targetLanguage = targetLanguageSelect[0].selectize.getValue();

        if (selectedAudioIndex === undefined || selectedAudioIndex === null || selectedAudioIndex === "") {
            showStatus('Please select an audio source.', 'warning');
            return;
        }

        if (!selectedDevice) {
            showStatus('Please select a processing device.', 'warning');
            return;
        }

        if (!sourceLanguage) {
            showStatus('Please select a source language.', 'warning');
            return;
        }

        if (!targetLanguage) {
            showStatus('Please select a target language.', 'warning');
            return;
        }

        eel.start_transcription(parseInt(selectedAudioIndex), selectedDevice, sourceLanguage, targetLanguage);
    });

    // Stop Button Click
    stopBtn.addEventListener('click', () => {
        eel.stop_transcription();
    });
});

// Eel functions to receive data from backend
eel.expose(receive_audio_sources);
function receive_audio_sources(sources) {
    const audioSourceSelect = document.getElementById('audioSource');
    audioSourceSelect.innerHTML = ''; // Clear existing options
    sources.forEach(source => {
        const option = document.createElement('option');
        option.value = source.index;
        option.textContent = source.label;
        audioSourceSelect.appendChild(option);
    });
}

eel.expose(receive_device_list);
function receive_device_list(devices) {
    const deviceSelector = document.getElementById('deviceSelector');
    deviceSelector.innerHTML = ''; // Clear existing options
    devices.forEach(device => {
        const option = document.createElement('option');
        option.value = device;
        option.textContent = device.toUpperCase();
        deviceSelector.appendChild(option);
    });
}

eel.expose(receive_supported_languages);
function receive_supported_languages(languages) {
    const sourceLanguageSelect = $('#sourceLanguage')[0].selectize;
    const targetLanguageSelect = $('#targetLanguage')[0].selectize;

    sourceLanguageSelect.clearOptions();
    targetLanguageSelect.clearOptions();

    languages.forEach(lang => {
        sourceLanguageSelect.addOption({value: lang, text: lang});
        targetLanguageSelect.addOption({value: lang, text: lang});
    });

    // Set placeholder options if no selection
    if (!sourceLanguageSelect.getValue()) {
        sourceLanguageSelect.setPlaceholder('Select source language...');
    }
    if (!targetLanguageSelect.getValue()) {
        targetLanguageSelect.setPlaceholder('Select target language...');
    }
}

eel.expose(receive_current_settings);
function receive_current_settings(settings) {
    const deviceSelector = document.getElementById('deviceSelector');
    const audioSourceSelect = document.getElementById('audioSource');
    const sourceLanguageSelect = $('#sourceLanguage')[0].selectize;
    const targetLanguageSelect = $('#targetLanguage')[0].selectize;

    // Set selected device
    if (settings.selected_device) {
        deviceSelector.value = settings.selected_device;
    }

    // Set selected audio source
    if (settings.selected_audio_source !== undefined && settings.selected_audio_source !== "") {
        audioSourceSelect.value = settings.selected_audio_source;
    }

    // Set selected languages
    if (settings.source_language) {
        sourceLanguageSelect.setValue(settings.source_language, true);
    }
    if (settings.target_language) {
        targetLanguageSelect.setValue(settings.target_language, true);
    }
}

eel.expose(update_raw_text);
function update_raw_text(text) {
    const rawTextArea = document.getElementById('rawText');
    rawTextArea.value += text + '\n';
    rawTextArea.scrollTop = rawTextArea.scrollHeight;
}

eel.expose(update_translated_text);
function update_translated_text(text) {
    const translatedTextArea = document.getElementById('translatedText');
    translatedTextArea.value += text + '\n';
    translatedTextArea.scrollTop = translatedTextArea.scrollHeight;
}

eel.expose(display_error);
function display_error(message) {
    showStatus(message, 'danger');
}

eel.expose(transcription_started);
function transcription_started() {
    document.getElementById('startBtn').disabled = true;
    document.getElementById('stopBtn').disabled = false;
    showStatus('Transcription started.', 'success');
}

eel.expose(transcription_stopped);
function transcription_stopped() {
    document.getElementById('startBtn').disabled = false;
    document.getElementById('stopBtn').disabled = true;
    showStatus('Transcription stopped.', 'secondary');
}

// Function to display status messages
function showStatus(message, type = 'info') {
    const statusBar = document.getElementById('statusBar');
    if (!statusBar) return;
    statusBar.className = ''; // Reset classes
    statusBar.classList.add('alert', `alert-${type}`, 'mt-3');
    statusBar.textContent = message;
    statusBar.classList.remove('d-none'); // Show the status bar
    // Automatically hide after 5 seconds
    setTimeout(() => {
        statusBar.classList.add('d-none');
    }, 5000);
}
