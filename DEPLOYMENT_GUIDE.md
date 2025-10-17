# AI助手应用部署上线教程 (Streamlit Cloud)

本文档将详细指导您如何将本项目部署到Streamlit Community Cloud，以获得一个公开、永久的网址，方便任何人访问。

---

## 部署流程总览

整个过程分为三步：

1.  **本地准备:** 在本地项目文件夹中创建Git代码仓库，并进行初次提交。
2.  **上传代码:** 在您的GitHub上创建一个新的**空**仓库，并将本地代码推送上去。
3.  **云端部署:** 登录Streamlit Community Cloud，关联您的GitHub仓库，配置密钥并一键部署。

---

## 步骤一：准备并上传代码到GitHub

*如果您已经将代码上传到GitHub，请跳到步骤二。*

1.  **初始化本地仓库:**
    在您电脑的项目文件夹中，打开终端，依次执行以下命令，将项目文件打包成一个版本。
    ```bash
    # 初始化一个新的Git仓库
    git init
    
    # 将所有文件添加到暂存区（.gitignore中指定的文件会被自动忽略）
    git add .
    
    # 创建第一个版本，并附上说明
    git commit -m "Initial commit"
    ```

2.  **在GitHub上创建新仓库:**
    *   登录 `github.com`。
    *   点击右上角的 **`+`** 号，选择 **`New repository`**。
    *   为仓库命名 (例如 `pvk-agent-app`)。
    *   权限选择 **`Public`** (这是Streamlit免费部署的前提)。
    *   **不要** 勾选任何 “Initialize this repository with:” 下方的选项。
    *   点击 **`Create repository`**。

3.  **关联并推送代码:**
    *   在GitHub创建完仓库的页面上，找到 `…or push an existing repository from the command line` 部分。
    *   复制下面的两行命令，并在您的本地终端中执行。这会将您的本地代码上传到GitHub。
    ```bash
    git remote add origin https://github.com/YourUsername/YourRepoName.git
    git push -u origin main
    ```

---

## 步骤二：在Streamlit Community Cloud上部署

1.  **登录Streamlit Cloud:**
    *   访问 [share.streamlit.io](https://share.streamlit.io) 并使用您的 **GitHub账号** 登录。

2.  **创建新应用 (New app):**
    *   登录后，点击 **`New app`** 按钮。
    *   在 `Repository` 下拉菜单中，选择您刚刚上传的仓库 (例如 `pvk-agent-app`)。
    *   确认 `Main file path` 填写的是 `app.py`。

3.  **配置密钥 (关键步骤):**
    *   点击 **`Advanced settings...`** 链接。
    *   在 **`Secrets`** 文本框中，**完整、精确地** 粘贴以下内容。请将 `sk-xxx...` 替换为您自己真实的DashScope API密钥。

    ```toml
    DASHSCOPE_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxx"
    ```
    *   **检查要点:** 密钥本身必须被英文双引号 `"` 包围。
    *   点击 **`Save`** 保存密钥。

4.  **部署:**
    *   点击蓝色的 **`Deploy!`** 按钮。

---

## 步骤三：部署成功

部署过程可能需要几分钟，Streamlit会自动为您安装 `requirements.txt` 中所有的库。完成后，您的应用就会上线，并且您会得到一个 `*.streamlit.app` 的公开网址。

## 如何更新应用

未来如果您修改了代码，只需将新的修改 `push` 到您GitHub仓库的 `main` 分支，Streamlit Cloud就会 **自动** 拉取最新代码并重新部署，应用会自动更新。
