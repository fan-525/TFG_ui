/**
 * 语音克隆管理模块
 * 统一处理语音克隆功能，提高代码复用性和可维护性
 */
class VoiceCloneManager {
    constructor() {
        this.apiEndpoints = {
            uploadAudio: '/api/upload-reference-audio',
            getRefAudios: '/api/reference-audios',
            voiceClone: '/audio_clone',
            getClonedAudios: '/api/cloned-audios'
        };
        this.progressCallbacks = new Map();
        this.uploadCallbacks = new Map();
    }

    /**
     * 上传音频文件
     * @param {File} file - 音频文件
     * @param {Function} callback - 回调函数 (progress, status, result)
     */
    async uploadAudioFile(file, callback) {
        if (!file) {
            throw new Error('请选择音频文件');
        }

        const formData = new FormData();
        formData.append('audio', file);

        try {
            callback?.(0, 'uploading', null);

            const response = await fetch(this.apiEndpoints.uploadAudio, {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            callback?.(100, 'completed', result);
            return result;

        } catch (error) {
            callback?.(0, 'error', null);
            throw new Error(`上传音频失败: ${error.message}`);
        }
    }

    /**
     * 获取参考音频列表
     * @returns {Promise<Array>} 参考音频列表
     */
    async getReferenceAudios() {
        try {
            const response = await fetch(this.apiEndpoints.getRefAudios);
            const result = await response.json();

            if (result.status === 'success' && result.files) {
                return result.files;
            } else {
                throw new Error(result.message || '获取参考音频列表失败');
            }
        } catch (error) {
            console.error('获取参考音频列表失败:', error);
            throw error;
        }
    }

    /**
     * 执行语音克隆
     * @param {Object} params - 克隆参数
     * @param {string} params.refAudioPath - 参考音频路径
     * @param {string} params.generateText - 生成文本
     * @param {string} params.outputFilename - 输出文件名（可选）
     * @param {Function} callback - 进度回调
     * @returns {Promise<Object>} 克隆结果
     */
    async performVoiceClone(params, callback) {
        const { refAudioPath, generateText, outputFilename } = params;

        // 验证必要参数
        if (!refAudioPath || !generateText) {
            throw new Error('请选择参考音频并输入生成文本');
        }

        const formData = new FormData();
        formData.append('ref_audio_path', refAudioPath);
        formData.append('generate_text', generateText);

        if (outputFilename) {
            formData.append('output_filename', outputFilename);
        }

        // 模拟进度更新
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress > 90) progress = 90;
            callback?.(progress, this.getProgressStatus(progress), null);
        }, 500);

        try {
            callback?.(0, 'processing', null);

            const response = await fetch(this.apiEndpoints.voiceClone, {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            clearInterval(progressInterval);
            callback?.(100, 'completed', result);

            return result;

        } catch (error) {
            clearInterval(progressInterval);
            callback?.(0, 'error', null);
            throw new Error(`语音克隆失败: ${error.message}`);
        }
    }

    /**
     * 获取进度状态文本
     * @param {number} progress - 进度百分比
     * @returns {string} 状态文本
     */
    getProgressStatus(progress) {
        if (progress < 20) return '分析参考音频...';
        if (progress < 40) return '处理文本内容...';
        if (progress < 60) return '进行语音克隆...';
        if (progress < 80) return '音频后处理...';
        return '完成处理...';
    }

    /**
     * 创建音频播放器
     * @param {string} audioPath - 音频路径
     * @param {string} title - 播放器标题
     * @param {Object} options - 可选参数
     * @returns {string} HTML字符串
     */
    createAudioPlayer(audioPath, title = '音频播放器', options = {}) {
        const {
            autoplay = false,
            controls = true,
            showPath = true,
            customClass = ''
        } = options;

        return `
            <div class="audio-player ${customClass}">
                <h3>${title}</h3>
                <audio controls ${autoplay ? 'autoplay' : ''} class="audio-element">
                    <source src="/${audioPath}" type="audio/wav">
                    <source src="/${audioPath.replace('.wav', '.mp3')}" type="audio/mpeg">
                    <source src="/${audioPath.replace('.wav', '.ogg')}" type="audio/ogg">
                    您的浏览器不支持音频播放
                </audio>
                ${showPath ? `<div class="audio-path">📁 ${audioPath}</div>` : ''}
            </div>
        `;
    }

    /**
     * 创建成功提示
     * @param {string} title - 标题
     * @param {string} message - 消息
     * @param {Object} details - 详细信息
     * @returns {string} HTML字符串
     */
    createSuccessAlert(title, message, details = {}) {
        return `
            <div class="success-alert">
                <div class="success-icon">✅</div>
                <div class="success-content">
                    <div class="success-title">${title}</div>
                    <div class="success-message">${message}</div>
                    ${details.generationTime ? `<div class="success-detail">⏱️ 耗时: ${details.generationTime}</div>` : ''}
                    ${details.text ? `<div class="success-detail">📝 文本: "${details.text}"</div>` : ''}
                    ${details.filePath ? `<div class="success-detail">📁 文件: ${details.filePath}</div>` : ''}
                </div>
            </div>
        `;
    }

    /**
     * 创建错误提示
     * @param {string} message - 错误消息
     * @returns {string} HTML字符串
     */
    createErrorAlert(message) {
        return `
            <div class="error-alert">
                <div class="error-icon">❌</div>
                <div class="error-content">
                    <div class="error-title">操作失败</div>
                    <div class="error-message">${message}</div>
                </div>
            </div>
        `;
    }

    /**
     * 创建进度条
     * @param {string} id - 进度条ID
     * @param {boolean} showPercentage - 是否显示百分比
     * @returns {string} HTML字符串
     */
    createProgressBar(id, showPercentage = true) {
        return `
            <div class="progress-container" id="${id}">
                <div class="progress-info">
                    <span class="progress-status">处理中...</span>
                    ${showPercentage ? '<span class="progress-percent">0%</span>' : ''}
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: 0%"></div>
                </div>
            </div>
        `;
    }

    /**
     * 更新进度条
     * @param {string} progressId - 进度条ID
     * @param {number} progress - 进度百分比
     * @param {string} status - 状态文本
     */
    updateProgressBar(progressId, progress, status) {
        const progressContainer = document.getElementById(progressId);
        if (!progressContainer) return;

        const progressFill = progressContainer.querySelector('.progress-fill');
        const progressStatus = progressContainer.querySelector('.progress-status');
        const progressPercent = progressContainer.querySelector('.progress-percent');

        if (progressFill) {
            progressFill.style.width = `${progress}%`;
        }

        if (progressStatus) {
            progressStatus.textContent = status;
        }

        if (progressPercent) {
            progressPercent.textContent = `${Math.round(progress)}%`;
        }
    }

    /**
     * 创建音频选择框
     * @param {string} id - 选择框ID
     * @param {string} className - CSS类名
     * @param {string} placeholder - 占位符文本
     * @returns {string} HTML字符串
     */
    createAudioSelect(id, className = '', placeholder = '请选择音频') {
        return `
            <select class="audio-select ${className}" id="${id}">
                <option value="">${placeholder}</option>
            </select>
        `;
    }

    /**
     * 填充音频选择框
     * @param {string} selector - 选择器
     * @param {Array} audios - 音频列表
     * @param {string} placeholder - 占位符文本
     */
    fillAudioSelect(selector, audios, placeholder = '请选择音频') {
        const selectElement = document.querySelector(selector);
        if (!selectElement) return;

        selectElement.innerHTML = `<option value="">${placeholder}</option>`;

        audios.forEach(audio => {
            const option = document.createElement('option');
            option.value = audio.relative_path || audio.path || audio.id;
            option.textContent = `${audio.filename} (${audio.size_mb}MB)`;
            option.title = `创建时间: ${audio.created_at}\n大小: ${audio.size_mb}MB`;
            selectElement.appendChild(option);
        });
    }

    /**
     * 创建刷新按钮
     * @param {string} id - 按钮ID
     * @param {string} text - 按钮文本
     * @param {string} className - CSS类名
     * @returns {string} HTML字符串
     */
    createRefreshButton(id, text = '刷新', className = '') {
        return `
            <button type="button" class="refresh-btn neon-btn-sm ${className}" id="${id}" title="刷新列表">
                ${text}
            </button>
        `;
    }

    /**
     * 处理按钮状态更新
     * @param {HTMLElement} button - 按钮元素
     * @param {string} state - 状态: 'loading', 'success', 'error', 'default'
     * @param {string} text - 按钮文本
     */
    updateButtonState(button, state, text) {
        if (!button) return;

        button.disabled = state !== 'default';
        button.innerHTML = text;

        // 移除所有状态类
        button.classList.remove('btn-loading', 'btn-success', 'btn-error');

        // 添加状态类
        if (state !== 'default') {
            button.classList.add(`btn-${state}`);
        }

        // 设置自动恢复
        if (state === 'success' || state === 'error') {
            setTimeout(() => {
                button.disabled = false;
                button.classList.remove('btn-success', 'btn-error');
                button.innerHTML = text === '刷新' ? text : '操作';
            }, 2000);
        }
    }

    /**
     * 显示通知消息
     * @param {string} message - 消息内容
     * @param {string} type - 消息类型: 'success', 'error', 'info', 'warning'
     * @param {number} duration - 显示时长（毫秒）
     */
    showNotification(message, type = 'info', duration = 3000) {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <span class="notification-icon">${this.getNotificationIcon(type)}</span>
            <span class="notification-message">${message}</span>
        `;

        document.body.appendChild(notification);

        // 显示动画
        setTimeout(() => {
            notification.classList.add('show');
        }, 10);

        // 自动移除
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        }, duration);
    }

    /**
     * 获取通知图标
     * @param {string} type - 通知类型
     * @returns {string} 图标
     */
    getNotificationIcon(type) {
        const icons = {
            info: 'ℹ️',
            success: '✅',
            warning: '⚠️',
            error: '❌'
        };
        return icons[type] || icons.info;
    }
}

// 导出模块
window.VoiceCloneManager = VoiceCloneManager;

// 创建全局实例
window.voiceCloneManager = new VoiceCloneManager();

// 导出工具函数
window.VoiceCloneUtils = {
    /**
     * 创建标准的三步语音克隆界面
     * @param {Object} config - 配置参数
     * @param {string} config.containerId - 容器ID
     * @param {string} config.uploadTitle - 上传标题
     * @param {string} config.selectTitle - 选择标题
     * @param {string} config.cloneTitle - 克隆标题
     * @param {Function} config.onCloneComplete - 克隆完成回调
     */
    createVoiceCloneInterface(config = {}) {
        const {
            containerId = 'voiceCloneContainer',
            uploadTitle = '步骤 1: 上传参考音频',
            selectTitle = '步骤 2: 选择参考音频',
            cloneTitle = '步骤 3: 语音克隆',
            onCloneComplete = null
        } = config;

        const container = document.getElementById(containerId);
        if (!container) {
            throw new Error(`容器元素不存在: ${containerId}`);
        }

        // 生成界面HTML
        container.innerHTML = `
            <div class="voice-clone-interface">
                <!-- 上传音频步骤 -->
                <div class="clone-step" data-step="1">
                    <div class="step-indicator">
                        <div class="step-number">1</div>
                        <div class="step-content">
                            <div class="step-title">${uploadTitle}</div>
                            <div class="step-description">上传您想要克隆的语音音频文件</div>
                        </div>
                    </div>
                    <div class="step-content">
                        <div class="upload-area">
                            <input type="file" class="audio-file-input" id="audioFileInput" accept=".wav,.mp3,.m4a,.flac,.ogg">
                            <button type="button" class="upload-btn neon-btn-primary" id="uploadBtn" disabled>
                                上传音频
                            </button>
                            <div class="upload-status" id="uploadStatus"></div>
                        </div>
                    </div>
                </div>

                <!-- 选择音频步骤 -->
                <div class="clone-step" data-step="2">
                    <div class="step-indicator">
                        <div class="step-number">2</div>
                        <div class="step-content">
                            <div class="step-title">${selectTitle}</div>
                            <div class="step-description">从已上传的音频中选择参考音频进行克隆</div>
                        </div>
                    </div>
                    <div class="step-content">
                        <div class="audio-select-area">
                            <select class="audio-select" id="refAudioSelect"></select>
                            <button type="button" class="refresh-btn neon-btn-sm" id="refreshRefAudioBtn" title="刷新音频列表">
                                刷新
                            </button>
                        </div>
                        <div class="audio-player-container" id="refAudioPlayer"></div>
                    </div>
                </div>

                <!-- 语音克隆步骤 -->
                <div class="clone-step" data-step="3">
                    <div class="step-indicator">
                        <div class="step-number">3</div>
                        <div class="step-content">
                            <div class="step-title">${cloneTitle}</div>
                            <div class="step-description">输入要生成的文本内容，进行语音克隆</div>
                        </div>
                    </div>
                    <div class="step-content">
                        <div class="clone-controls">
                            <div class="text-input-area">
                                <textarea
                                    class="clone-text-input"
                                    id="cloneTextInput"
                                    placeholder="请输入您想要生成的语音文本内容..."
                                    rows="4"
                                ></textarea>
                            </div>
                            <div class="optional-controls">
                                <input type="text"
                                       class="output-filename-input"
                                       id="outputFilenameInput"
                                       placeholder="输出文件名（可选）">
                            </div>
                            <button type="button" class="clone-btn neon-btn-primary" id="startCloneBtn">
                                开始语音克隆
                            </button>
                            <div class="progress-container" id="cloneProgress" style="display: none;">
                                <div class="progress-info">
                                    <span class="progress-status">克隆中...</span>
                                    <span class="progress-percent">0%</span>
                                </div>
                                <div class="progress-bar">
                                    <div class="progress-fill" style="width: 0%"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 结果展示区域 -->
                <div class="clone-result" id="cloneResult">
                    <div class="result-placeholder">
                        <div class="placeholder-icon">🎤</div>
                        <h3>语音克隆结果</h3>
                        <p>完成语音克隆后，结果将在此显示</p>
                    </div>
                </div>
            </div>
        `;

        // 绑一事件处理
        this.setupEventListeners(config);
    },

    /**
     * 设置事件监听器
     * @param {Object} config - 配置参数
     */
    setupEventListeners(config) {
        // 文件选择事件
        const fileInput = document.getElementById('audioFileInput');
        if (fileInput) {
            fileInput.addEventListener('change', (e) => {
                const file = e.target.files[0];
                const uploadBtn = document.getElementById('uploadBtn');
                const uploadStatus = document.getElementById('uploadStatus');

                if (file) {
                    uploadBtn.disabled = false;
                    uploadBtn.textContent = `上传 ${file.name}`;
                    uploadStatus.textContent = `已选择: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)}MB)`;
                    uploadStatus.style.color = 'var(--neon-green)';
                } else {
                    uploadBtn.disabled = true;
                    uploadBtn.textContent = '上传音频';
                    uploadStatus.textContent = '';
                }
            });
        }

        // 上传事件
        const uploadBtn = document.getElementById('uploadBtn');
        if (uploadBtn) {
            uploadBtn.addEventListener('click', async () => {
                const file = fileInput.files[0];
                if (!file) {
                    window.voiceCloneManager.showNotification('请先选择要上传的音频文件', 'warning');
                    return;
                }

                const uploadBtn = document.getElementById('uploadBtn');
                const uploadStatus = document.getElementById('uploadStatus');

                try {
                    window.voiceCloneManager.updateButtonState(uploadBtn, 'loading', '上传中...');

                    const result = await window.voiceCloneManager.uploadAudioFile(file, (progress, status, result) => {
                        if (status === 'uploading') {
                            uploadStatus.textContent = `上传中... ${Math.round(progress)}%`;
                        } else if (status === 'completed') {
                            uploadStatus.textContent = result.status === 'success' ?
                                `✅ 上传成功: ${result.filename}` :
                                `❌ 上传失败: ${result.message}`;
                        }
                    });

                    if (result.status === 'success') {
                        // 刷新音频列表
                        await this.loadReferenceAudios();

                        // 清空文件选择
                        fileInput.value = '';
                        uploadBtn.disabled = true;
                        uploadBtn.textContent = '上传音频';

                        window.voiceCloneManager.showNotification('音频上传成功', 'success');
                    }

                } catch (error) {
                    uploadStatus.textContent = '❌ 上传失败';
                    uploadStatus.style.color = 'var(--neon-red)';
                    window.voiceCloneManager.showNotification(`上传音频失败: ${error.message}`, 'error');
                } finally {
                    if (uploadBtn.innerHTML === '上传中...') {
                        window.voiceCloneManager.updateButtonState(uploadBtn, 'default', '上传音频');
                    }
                }
            });
        }

        // 刷新音频列表事件
        const refreshBtn = document.getElementById('refreshRefAudioBtn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', async () => {
                await this.loadReferenceAudios();
            });
        }

        // 加载音频事件
        const refAudioSelect = document.getElementById('refAudioSelect');
        if (refAudioSelect) {
            refAudioSelect.addEventListener('change', async (e) => {
                const audioPath = e.target.value;
                if (audioPath) {
                    await this.loadAudioPlayer(audioPath);
                }
            });
        }

        // 开始克隆事件
        const startCloneBtn = document.getElementById('startCloneBtn');
        if (startCloneBtn) {
            startCloneBtn.addEventListener('click', async () => {
                await this.startVoiceClone(config);
            });
        }
    },

    /**
     * 加载参考音频列表
     */
    async loadReferenceAudios() {
        const refAudioSelect = document.getElementById('refAudioSelect');
        const refreshBtn = document.getElementById('refreshRefAudioBtn');

        if (refreshBtn) {
            window.voiceCloneManager.updateButtonState(refreshBtn, 'loading', '刷新中...');
        }

        try {
            const audios = await window.voiceCloneManager.getReferenceAudios();
            window.voiceCloneManager.fillAudioSelect('#refAudioSelect', audios);

            if (refreshBtn) {
                window.voiceCloneManager.updateButtonState(refreshBtn, 'success', '已刷新');
            }
        } catch (error) {
            console.error('加载参考音频列表失败:', error);
            if (refreshBtn) {
                window.voiceCloneManager.updateButtonState(refreshBtn, 'error', '刷新失败');
            }
            window.voiceCloneManager.showNotification('加载音频列表失败', 'error');
        }
    },

    /**
     * 加载音频播放器
     * @param {string} audioPath - 音频路径
     */
    async loadAudioPlayer(audioPath) {
        const playerContainer = document.getElementById('refAudioPlayer');
        if (!playerContainer) return;

        try {
            playerContainer.innerHTML = window.voiceCloneManager.createAudioPlayer(
                audioPath, '参考音频播放器', { showPath: true }
            );

            const audio = playerContainer.querySelector('.audio-element');
            audio.load();

            audio.addEventListener('error', () => {
                playerContainer.innerHTML = `
                    <div class="error-placeholder">
                        <h3>❌ 音频加载失败</h3>
                        <p>请检查音频路径是否正确</p>
                        <small>${audioPath}</small>
                    </div>
                `;
            });

            audio.addEventListener('loadeddata', () => {
                console.log('参考音频加载成功:', audioPath);
            });

        } catch (error) {
            console.error('加载音频播放器失败:', error);
            playerContainer.innerHTML = `
                <div class="error-placeholder">
                    <h3>❌ 音频播放器创建失败</h3>
                    <p>${error.message}</p>
                </div>
            `;
        }
    },

    /**
     * 开始语音克隆
     * @param {Object} config - 配置参数
     */
    async startVoiceClone(config) {
        const refAudioPath = document.getElementById('refAudioSelect').value.trim();
        const generateText = document.getElementById('cloneTextInput').value.trim();
        const outputFilename = document.getElementById('outputFilenameInput').value.trim();
        const startBtn = document.getElementById('startCloneBtn');
        const progressContainer = document.getElementById('cloneProgress');
        const resultContainer = document.getElementById('cloneResult');

        if (!refAudioPath || !generateText) {
            window.voiceCloneManager.showNotification('请选择参考音频并输入生成文本', 'warning');
            return;
        }

        try {
            progressContainer.style.display = 'block';
            window.voiceCloneManager.updateButtonState(startBtn, 'loading', '克隆中...');

            const result = await window.voiceCloneManager.performVoiceClone({
                refAudioPath,
                generateText,
                outputFilename
            }, (progress, status, result) => {
                window.voiceCloneManager.updateProgressBar('cloneProgress', progress,
                    window.voiceCloneManager.getProgressStatus(progress));
            });

            if (result.status === 'success) {
                progressContainer.style.display = 'none';
                window.voiceCloneManager.updateButtonState(startBtn, 'success', '克隆成功');

                // 显示结果
                if (result.cloned_audio_path) {
                    const details = {
                        generationTime: result.generation_time ?
                            `${result.generation_time.toFixed(2)}秒` : null,
                        text: generateText,
                        filePath: result.cloned_audio_path
                    };

                    resultContainer.innerHTML = `
                        ${window.voiceCloneManager.createSuccessAlert('语音克隆成功', '语音克隆成功！', details)}
                        ${window.voiceCloneManager.createAudioPlayer(
                            result.cloned_audioPath,
                            '克隆音频播放器',
                            { autoplay: true, showPath: true }
                        )}
                    `;
                }

                // 调用完成回调
                if (config.onCloneComplete) {
                    config.onCloneComplete(result);
                }

                window.voiceCloneManager.showNotification('语音克隆成功！', 'success');
            } else {
                progressContainer.style.display = 'none';
                window.voiceCloneManager.updateButtonState(startBtn, 'error', '克隆失败');

                // 显示错误
                resultContainer.innerHTML = `
                    ${window.voiceCloneManager.createErrorAlert(result.message || '语音克隆失败')}
                `;

                window.voiceCloneManager.showNotification(`语音克隆失败: ${result.message || '未知错误'}`, 'error');
            }

        } catch (error) {
            progressContainer.style.display = 'none';
            window.voiceCloneManager.updateButtonState(startBtn, 'error', '克隆失败');

            // 显示错误
            resultContainer.innerHTML = `
                ${window.voiceCloneManager.createErrorAlert(error.message)}
            `;

            window.voiceCloneManager.showNotification(`语音克隆失败: ${error.message}`, 'error');
        } finally {
            setTimeout(() => {
                window.voiceCloneManager.updateButtonState(startBtn, 'default', '开始语音克隆');
            }, 2000);
        }
    }
};