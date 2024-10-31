# Introduction

TODO: Give a short introduction of your project. Let this section explain the objectives or the motivation behind this project.

# Getting Started

1. Installation process

   - Clone the repository
   - Create a conda virtual environment by using the following command

   ```
   conda create -f conda_env.yml -y --quiet
   ```

   - Install the required packages by using the following command

   ```
    pip install -r requirements.txt
   ```

   - Install the requests and langchain-openai package separately by using the following command due to conflicting requirements

   ```
   pip install langchain-openai==0.1.23
   pip install requests==2.27.1

   ```

# Build and Test

TODO: Describe and show how to build your code and run the tests.

# Contribute

TODO: Explain how other users and developers can contribute to make your code better.

If you want to learn more about creating good readme files then refer the following [guidelines](https://docs.microsoft.com/en-us/azure/devops/repos/git/create-a-readme?view=azure-devops). You can also seek inspiration from the below readme files:

- [ASP.NET Core](https://github.com/aspnet/Home)
- [Visual Studio Code](https://github.com/Microsoft/vscode)
- [Chakra Core](https://github.com/Microsoft/ChakraCore)
