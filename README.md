# AI-Code-Obfuscator: 智能代码去重与反抄袭工具
An intelligent code anti-plagiarism tool for C/C++/Python that uses AI (LLM) to transform variable/function names into meaningful equivalents and modify code structure.

---

## 🚀 项目概述 (Overview)

**AI-Code-Obfuscator** 是一个用于代码**反抄袭（Anti-Plagiarism）**和**去重（Deduplication）**的命令行工具。

传统的混淆工具通常使用随机或无意义的名称替换变量，而本项目利用 **大型语言模型 (LLM)** 的能力，在重命名变量和函数时，**分析其上下文逻辑，并生成具有实际意义的新名称**。这不仅能有效改变代码特征，还能同时提高代码的可读性，非常适用于代码作业提交、知识产权保护等场景。

## ✨ 主要特点 (Features)

* **🧠 AI 智能命名：** 集成 Gemini 或 OpenAI API，根据代码上下文分析变量和函数用途，生成更具描述性的名称。
* **🛡️ 跨语言支持：** 完整支持 **Python**、**C** 和 **C++** 语言的转换和处理。
* **🐍 Python AST 转换：** 使用 `ast` 和 `astor` 库进行精确的 Python 变量和函数名重命名。
* **🧱 C/C++ 结构变化：** 支持 C/C++ 变量的智能重命名，并进行格式（如大括号风格）、函数顺序的随机变化。
* **✍️ 智能注释生成：** 在关键函数和代码块前添加有意义的注释。
* **🔑 API 配置化：** 支持通过 `config.json` 文件灵活配置 API Key、URL 和 LLM 模型名称。

## ⚙️ 环境要求与安装 (Installation)

本项目运行需要 Python 3 环境，并依赖 `requests` 和 `astor` 库。

### 1. 克隆仓库

```bash
git clone [https://github.com/YourUsername/AI-Code-Obfuscator.git](https://github.com/YourUsername/AI-Code-Obfuscator.git)
cd AI-Code-Obfuscator
````

### 2\. 安装 Python 依赖

```bash
pip install requests astor
```

### 3\. API 配置（关键步骤）

要启用 AI 智能命名功能，您需要配置 API 密钥：

1.  **运行程序**：
    ```bash
    python code_obfuscator.py
    ```
2.  当程序提示 **“No valid API configuration found.”** 时，输入 `y` 创建 `config.json` 模板。
3.  **获取 API Key：** 访问 [Google AI Studio](https://makersuite.google.com/app/apikey) 获取您的 **Gemini API Key**。
4.  **编辑 `config.json`：**
      * 将 `"api_key"` 替换为您的密钥。
      * 推荐模型设置为 `"gemini-1.5-flash"`（或 `"gemini-1.5-pro"`）。
      * 将 `"api_url"` 留空或删除。

## 📚 使用指南 (Usage Guide)

### 1\. 运行程序

```bash
python code_obfuscator.py
```

### 2\. 输入参数

程序将依次提示您输入以下信息：

| 提示 | 示例输入 | 描述 |
| :--- | :--- | :--- |
| `Enter the path to your code file:` | `main.py` 或 `source.cpp` | 要处理的源代码文件的路径。 |
| `Enter the programming language (c/cpp/python):` | `python` | 代码的语言类型。 |
| `Enter output file path (press Enter for auto):` | `output.py` 或 **(留空)** | 输出文件路径。留空则自动生成 `[文件名]_modified.[ext]`。 |

### 3\. 查看结果

程序运行完成后，将在终端显示重命名变量的总数，并将修改后的代码保存到指定的输出文件中。

## 🛠️ 核心模块 (Core Modules)

  * **`AICodeAnalyzer`:** 负责与 LLM API 通信，根据代码上下文获取变量和函数的有意义名称建议。
  * **`CodeAntiPlagiarism`:** 主处理器类，负责读取文件、选择语言处理流程（Python 或 C/C++）和保存结果。
  * **`PythonTransformer`:** 继承 `ast.NodeTransformer`，用于对 Python 抽象语法树进行精确的代码修改。

## 🤝 贡献与致谢 (Contribution & Acknowledgement)

欢迎提交 Issue 或 Pull Request 来改进本项目。

-----
