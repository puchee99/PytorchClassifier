<div id="top"></div>

<!-- PROJECT SHIELDS -->
<!--
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/puchee99/PytorchClassifier">
    <img src="images/pytorch.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">PytorchClassifier</h3>

  <p align="center">
    Given a csv with a target column, it processes the data and trains a predictor using neural networks.
    <br />
    <a href="https://github.com/puchee99/PytorchClassifier"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/puchee99/PytorchClassifier">View Demo</a>
    ·
    <a href="https://github.com/puchee99/PytorchClassifier/issues">Report Bug</a>
    ·
    <a href="https://github.com/puchee99/PytorchClassifier/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

![product-screenshot]

[Image][product-screenshot]

The goal of this project is to classify data using neural networks. It should be good enough to be cut up and used for different projects.

<p align="right">(<a href="#top">back to top</a>)</p>



### Built With

* [Pytorch](https://pytorch.org/)
* [scikit-learn](https://scikit-learn.org/)
* [Numpy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [Logging](https://docs.python.org/3/library/logging.html)
* [Seaborn](https://seaborn.pydata.org/)

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

Given a csv with a target column, it processes the data and trains a predictor using neural networks.


### Installation


First, clone the repository:
   ```sh
   git clone https://github.com/puchee99/PytorchClassifier.git
   ```
Access to the project folder with:
  ```sh
  cd PytorchClassifier
  ```

We will create a virtual environment with `python3`
* Create environment with python 3 
    ```sh
    python3 -m venv venv
    ```
    
* Enable the virtual environment
    ```sh
    source venv/bin/activate
    ```

* Install the python dependencies on the virtual environment
    ```sh
    pip install -r requirements.txt
    ```

<p align="right">(<a href="#top">back to top</a>)</p>

## Usage

Run `python train.py`


## Roadmap

- [x] Train model
- [x] Loggers
- [ ] BI
    - [ ] Flask
    - [ ] Plotly

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- CONTACT -->
## Contact

Arnau Puche  - [@arnau_puche_vila](https://www.linkedin.com/in/arnau-puche-vila-ds/) - arnaupuchevila@gmail.com

Project Link: [https://github.com/puchee99/PytorchClassifier](https://github.com/puchee99/PytorchClassifier)


<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/puchee99/PytorchClassifier.svg?style=for-the-badge
[contributors-url]: https://github.com/puchee99/PytorchClassifier/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/puchee99/PytorchClassifier.svg?style=for-the-badge
[forks-url]: https://github.com/puchee99/PytorchClassifier/network/members
[stars-shield]: https://img.shields.io/github/stars/puchee99/PytorchClassifier.svg?style=for-the-badge
[stars-url]: https://github.com/puchee99/PytorchClassifier/stargazers
[issues-shield]: https://img.shields.io/github/issues/puchee99/PytorchClassifier.svg?style=for-the-badge
[issues-url]: https://github.com/puchee99/PytorchClassifier/issues
[license-shield]: https://img.shields.io/github/license/puchee99/PytorchClassifier.svg?style=for-the-badge
[license-url]: https://github.com/puchee99/PytorchClassifier/blob/main/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/arnau-puche-vila-ds/
[product-screenshot]: images/figures.png
