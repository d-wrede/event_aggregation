# Event Aggregation Project

This repository hosts an event aggregation project tailored for Telegram channels, where it parses event invitations within the messages, extracts relevant information (like date and topic), and compiles an organized summary. The output is then posted on the same channel in a chronological format, enhancing readability and accessibility, particularly for channels that have high frequencies of event announcements.

The project currently supports the German language, but it is designed to be language-agnostic as long as language packages for the underlying algorithms are available, making it adaptable to diverse linguistic contexts.

The extraction of date and timestamp information is performed using regular expressions due to their precision and reliability, providing specific details such as day, month, year, and time. We had experimented with named entity recognition using the Spacy library, but we found it to be less reliable and precise, particularly for non-English/German languages. Please note that this file is not included in the public repository.

Topic extraction is carried out using a combination of NLP algorithms. The parameters for these algorithms, including those for scoring the combination, are fine-tuned using the "Covariance Matrix Adaptation Evolution Strategy" (CMA-ES). This aspect is presently the main focus of our project: to create a stable parameter optimization process based on the "chosen_topics.json" comparison file.

## Roadmap

Looking ahead, our upcoming development plans include:

1. Finalize the optimization process and ensure effective topic extraction.
2. Enable the system to send an event overview as a message to the Telegram channel, which will:
   1. List event announcements in chronological order, complete with date, topic, and a link to the original message.
3. Automate the system for regular checks and updates of event announcements.
4. Enhance the Telegram bot's functionality to enable invitations to groups and perform requisite actions within these groups.
5. Utilize AWS CloudWatch to schedule and manage regular runs of the project.

## License Information

This project is licensed under the Apache 2.0 License.

## Contact Information

Should you have any queries or need assistance, feel free to reach out:

- Portfolio website: [www.daniel-wrede.de](http://www.daniel-wrede.de)
- LinkedIn: [Daniel Wrede](https://www.linkedin.com/in/danielwrede/)
- Email: projects@daniel-wrede.de

We aim to make our README as comprehensive and user-friendly as possible, attracting more users and contributors to the project. Your feedback and contributions are always welcome!