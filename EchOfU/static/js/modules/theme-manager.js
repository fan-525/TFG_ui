/**
 * 主题管理模块
 * 统一处理浅色/深色模式切换
 */
class ThemeManager {
    constructor() {
        this.currentTheme = 'light';
        this.storageKey = 'synctalk-theme';
        this.transitionDuration = 300;
        this.isTransitioning = false;
        this.observers = [];
        this.mediaQuery = null;

        this.init();
    }

    /**
     * 初始化主题管理器
     */
    init() {
        // 从localStorage恢复主题设置
        this.loadThemeFromStorage();

        // 监听系统主题变化
        this.initSystemThemeDetection();

        // 监听页面加载
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => {
                this.applyTheme(this.currentTheme);
                this.setupThemeToggles();
            });
        } else {
            this.applyTheme(this.currentTheme);
            this.setupThemeToggles();
        }

        // 监听键盘快捷键
        this.setupKeyboardShortcuts();

        // 添加页面可见性监听
        this.setupVisibilityDetection();
    }

    /**
     * 从localStorage加载主题
     */
    loadThemeFromStorage() {
        try {
            const saved = localStorage.getItem(this.storageKey);
            if (saved && (saved === 'light' || saved === 'dark')) {
                this.currentTheme = saved;
            } else {
                // 如果没有保存的主题，检测系统主题
                this.currentTheme = this.getSystemTheme();
            }
        } catch (error) {
            console.warn('无法从localStorage加载主题设置:', error);
            this.currentTheme = this.getSystemTheme();
        }
    }

    /**
     * 获取系统主题
     */
    getSystemTheme() {
        if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
            return 'dark';
        }
        return 'light';
    }

    /**
     * 初始化系统主题检测
     */
    initSystemThemeDetection() {
        if (window.matchMedia) {
            this.mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');

            // 监听系统主题变化（仅在用户没有手动设置时）
            this.mediaQuery.addEventListener('change', (e) => {
                const hasManualTheme = localStorage.getItem(this.storageKey) !== null;
                if (!hasManualTheme) {
                    const systemTheme = e.matches ? 'dark' : 'light';
                    this.applyTheme(systemTheme);
                    this.currentTheme = systemTheme;
                }
            });
        }
    }

    /**
     * 设置主题切换按钮
     */
    setupThemeToggles() {
        // 查找所有主题切换按钮
        const toggles = document.querySelectorAll('[data-theme-toggle]');

        toggles.forEach(toggle => {
            // 设置初始状态
            this.updateToggleButton(toggle, this.currentTheme);

            // 添加点击事件
            toggle.addEventListener('click', (e) => {
                e.preventDefault();
                this.toggleTheme();
            });
        });
    }

    /**
     * 更新主题切换按钮状态
     */
    updateToggleButton(toggle, theme) {
        const isActive = theme === 'dark';

        // 更新active类
        if (isActive) {
            toggle.classList.add('active');
        } else {
            toggle.classList.remove('active');
        }

        // 更新aria属性
        toggle.setAttribute('aria-pressed', isActive.toString());
        toggle.setAttribute('aria-label', isActive ? '切换到浅色模式' : '切换到深色模式');

        // 更新tooltip
        const tooltip = toggle.querySelector('.theme-tooltip');
        if (tooltip) {
            tooltip.textContent = isActive ? '切换到浅色模式' : '切换到深色模式';
        }
    }

    /**
     * 切换主题
     */
    toggleTheme() {
        if (this.isTransitioning) return;

        const newTheme = this.currentTheme === 'light' ? 'dark' : 'light';
        this.setTheme(newTheme);
    }

    /**
     * 设置主题
     */
    setTheme(theme) {
        if (!theme || (theme !== 'light' && theme !== 'dark')) {
            console.warn('无效的主题值:', theme);
            return;
        }

        if (this.currentTheme === theme) return;

        this.isTransitioning = true;

        // 添加过渡效果类
        document.documentElement.classList.add('theme-transitioning');

        // 应用主题
        this.applyTheme(theme);

        // 更新当前主题
        this.currentTheme = theme;

        // 保存到localStorage
        this.saveThemeToStorage(theme);

        // 更新所有切换按钮
        this.updateAllToggles(theme);

        // 通知观察者
        this.notifyObservers(theme);

        // 移除过渡效果类
        setTimeout(() => {
            document.documentElement.classList.remove('theme-transitioning');
            this.isTransitioning = false;
        }, this.transitionDuration);
    }

    /**
     * 应用主题到DOM
     */
    applyTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);

        // 设置meta标签
        this.updateMetaThemeColor(theme);

        // 触发自定义事件
        const event = new CustomEvent('themechange', {
            detail: {
                theme: theme,
                previousTheme: this.currentTheme
            }
        });
        document.dispatchEvent(event);
    }

    /**
     * 更新meta主题色
     */
    updateMetaThemeColor(theme) {
        let themeColor;

        if (theme === 'dark') {
            themeColor = '#0a0e1a';
        } else {
            themeColor = '#f7fbff';
        }

        // 更新或创建meta theme-color标签
        let metaThemeColor = document.querySelector('meta[name="theme-color"]');
        if (!metaThemeColor) {
            metaThemeColor = document.createElement('meta');
            metaThemeColor.name = 'theme-color';
            document.head.appendChild(metaThemeColor);
        }
        metaThemeColor.content = themeColor;

        // 更新msapplication-TileColor
        let metaTileColor = document.querySelector('meta[name="msapplication-TileColor"]');
        if (!metaTileColor) {
            metaTileColor = document.createElement('meta');
            metaTileColor.name = 'msapplication-TileColor';
            document.head.appendChild(metaTileColor);
        }
        metaTileColor.content = themeColor;
    }

    /**
     * 更新所有主题切换按钮
     */
    updateAllToggles(theme) {
        const toggles = document.querySelectorAll('[data-theme-toggle]');
        toggles.forEach(toggle => {
            this.updateToggleButton(toggle, theme);
        });
    }

    /**
     * 保存主题到localStorage
     */
    saveThemeToStorage(theme) {
        try {
            localStorage.setItem(this.storageKey, theme);
        } catch (error) {
            console.warn('无法保存主题设置到localStorage:', error);
        }
    }

    /**
     * 设置键盘快捷键
     */
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + Shift + T 切换主题
            if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'T') {
                e.preventDefault();
                this.toggleTheme();
            }

            // Ctrl/Cmd + Shift + D 切换到深色模式
            if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'D') {
                e.preventDefault();
                this.setTheme('dark');
            }

            // Ctrl/Cmd + Shift + L 切换到浅色模式
            if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'L') {
                e.preventDefault();
                this.setTheme('light');
            }
        });
    }

    /**
     * 设置页面可见性检测
     */
    setupVisibilityDetection() {
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden) {
                // 页面变为可见时，检查主题是否需要更新
                const storedTheme = localStorage.getItem(this.storageKey);
                if (storedTheme && storedTheme !== this.currentTheme) {
                    this.setTheme(storedTheme);
                }
            }
        });
    }

    /**
     * 添加主题变化观察者
     */
    addObserver(callback) {
        if (typeof callback === 'function') {
            this.observers.push(callback);
        }
    }

    /**
     * 移除主题变化观察者
     */
    removeObserver(callback) {
        const index = this.observers.indexOf(callback);
        if (index > -1) {
            this.observers.splice(index, 1);
        }
    }

    /**
     * 通知所有观察者
     */
    notifyObservers(theme) {
        this.observers.forEach(callback => {
            try {
                callback(theme, this.currentTheme);
            } catch (error) {
                console.warn('主题变化观察者回调错误:', error);
            }
        });
    }

    /**
     * 获取当前主题
     */
    getCurrentTheme() {
        return this.currentTheme;
    }

    /**
     * 获取下一个主题
     */
    getNextTheme() {
        return this.currentTheme === 'light' ? 'dark' : 'light';
    }

    /**
     * 重置为系统主题
     */
    resetToSystemTheme() {
        localStorage.removeItem(this.storageKey);
        const systemTheme = this.getSystemTheme();
        this.setTheme(systemTheme);
    }

    /**
     * 检查是否为深色模式
     */
    isDarkMode() {
        return this.currentTheme === 'dark';
    }

    /**
     * 检查是否为浅色模式
     */
    isLightMode() {
        return this.currentTheme === 'light';
    }

    /**
     * 获取主题信息
     */
    getThemeInfo() {
        return {
            current: this.currentTheme,
            system: this.getSystemTheme(),
            stored: localStorage.getItem(this.storageKey),
            isSystemDefault: localStorage.getItem(this.storageKey) === null,
            isTransitioning: this.isTransitioning,
            transitionDuration: this.transitionDuration
        };
    }

    /**
     * 销毁主题管理器
     */
    destroy() {
        this.observers = [];
        if (this.mediaQuery) {
            this.mediaQuery.removeEventListener('change', () => {});
        }
    }
}

// 创建全局主题管理器实例
window.themeManager = new ThemeManager();

// 导出工具函数
window.ThemeUtils = {
    /**
     * 切换主题
     */
    toggle: () => window.themeManager.toggleTheme(),

    /**
     * 设置主题
     */
    set: (theme) => window.themeManager.setTheme(theme),

    /**
     * 获取当前主题
     */
    getCurrent: () => window.themeManager.getCurrentTheme(),

    /**
     * 检查是否为深色模式
     */
    isDark: () => window.themeManager.isDarkMode(),

    /**
     * 检查是否为浅色模式
     */
    isLight: () => window.themeManager.isLightMode(),

    /**
     * 重置为系统主题
     */
    reset: () => window.themeManager.resetToSystemTheme(),

    /**
     * 添加主题变化监听器
     */
    onChange: (callback) => window.themeManager.addObserver(callback),

    /**
     * 移除主题变化监听器
     */
    offChange: (callback) => window.themeManager.removeObserver(callback)
};

// 导出类
window.ThemeManager = ThemeManager;