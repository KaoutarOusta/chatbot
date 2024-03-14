import pandas as pd
import json
from sklearn.model_selection import train_test_split

# Charger votre jeu de données
dataset = {
  "messages": [
    {"role": "user", "content": "Can you tell me about your website design options?"},
    {"role": "assistant", "content": "Certainly! We offer a range of website design options to suit your needs. We have a variety of templates and themes to choose from, and our platform allows for extensive customization to make your website unique."},

    {"role": "user", "content": "What sets your website creation service apart from others like Wix?"},
    {"role": "assistant", "content": "What sets us apart is our commitment to providing a seamless and user-friendly website creation experience. Our platform is known for its flexibility, powerful features, and outstanding customer support. We also offer competitive pricing and a wide range of tools for businesses of all sizes."},

    {"role": "user", "content": "How long does it typically take to create a website with your platform?"},
    {"role": "assistant", "content": "The time it takes to create a website can vary depending on the complexity of the project and your familiarity with our platform. However, many users can create a basic website in just a few hours, while more complex projects may take several days or weeks."},

    {"role": "user", "content": "Can I see some examples of websites created using your service?"},
    {"role": "assistant", "content": "Certainly! You can check out our portfolio of websites created by our users on our website. We showcase a diverse range of websites to demonstrate the versatility of our platform."},

    {"role": "user", "content": "What customization options are available for website templates?"},
    {"role": "assistant", "content": "Our platform offers extensive customization options. You can change colors, fonts, layouts, and add your own images and content. We also provide a drag-and-drop editor for easy customization, so you can make your website look exactly the way you want."},

    {"role": "user", "content": "Do you offer e-commerce capabilities for online stores?"},
    {"role": "assistant", "content": "Yes, we do! We have built-in e-commerce features that allow you to create and manage online stores. You can add products, set prices, and handle transactions securely through our platform."},

    {"role": "user", "content": "How do you handle website hosting and maintenance?"},
    {"role": "assistant", "content": "We offer reliable hosting services for your website, and our platform includes automated updates and maintenance to ensure your site runs smoothly. Our technical support team is also available to assist with any hosting or maintenance issues."},

    {"role": "user", "content": "What is the pricing structure for your website creation service?"},
    {"role": "assistant", "content": "Our pricing structure varies depending on the features and services you need. We offer both free and premium plans. You can find detailed pricing information on our website."},

    {"role": "user", "content": "Can I integrate third-party tools and plugins into websites built with your platform?"},
    {"role": "assistant", "content": "Yes, you can! Our platform supports integration with a wide range of third-party tools and plugins, allowing you to add additional functionality and features to your website."},

    {"role": "user", "content": "Is it possible to migrate an existing website to your platform?"},
    {"role": "assistant", "content": "Yes, we offer migration services to help you move your existing website to our platform. Our team can assist you with the process to ensure a smooth transition."},

    {"role": "user", "content": "Do you provide SEO and marketing features to help websites rank higher on search engines?"},
    {"role": "assistant", "content": "Yes, we offer SEO tools and marketing features to help improve the visibility of your website on search engines. You can optimize your content, meta tags, and more to boost your rankings."},

    {"role": "user", "content": "What kind of support and assistance do you offer to users while building their websites?"},
    {"role": "assistant", "content": "We provide comprehensive support throughout the website creation process. Our platform includes tutorials and guides, and our customer support team is available via chat and email to assist with any questions or issues you may have."},

    {"role": "user", "content": "Are there any limitations or restrictions on the types of websites I can create with your service?"},
    {"role": "assistant", "content": "While our platform is versatile, there may be some limitations based on your chosen plan. For specific details on limitations and restrictions, please refer to our terms of service and plan features on our website."},

    {"role": "user", "content": "Can I access and edit my website's code directly if needed?"},
    {"role": "assistant", "content": "Yes, you can! We provide access to the website's code for users who want to make advanced customizations. You have the flexibility to edit the code directly when necessary."},

    {"role": "user", "content": "What security measures do you have in place to protect websites built with your platform?"},
    {"role": "assistant", "content": "We take website security seriously. Our platform includes security features such as SSL certificates, regular backups, and protection against common vulnerabilities. We also monitor for suspicious activity to keep your website safe."},

    // Additional questions
    {"role": "user", "content": "Do you offer responsive design for mobile devices?"},
    {"role": "assistant", "content": "Yes, all websites created on our platform are designed to be responsive, ensuring they look and function well on both desktop and mobile devices."},

    {"role": "user", "content": "Can I transfer my domain to your platform?"},
    {"role": "assistant", "content": "Yes, you can transfer your existing domain to our platform, or you can register a new domain through our service. We provide domain management tools for your convenience."},

    {"role": "user", "content": "Are there any additional fees for using e-commerce features on my website?"},
    {"role": "assistant", "content": "While our e-commerce features are included in some plans, there may be transaction fees associated with payment processing. Be sure to review our pricing details for specific information."},

    {"role": "user", "content": "What payment gateways do you support for online transactions?"},
    {"role": "assistant", "content": "We support a variety of popular payment gateways, including PayPal, Stripe, and more. You can choose the one that best suits your business needs."},

    {"role": "user", "content": "Can I use my own custom domain name with your service?"},
    {"role": "assistant", "content": "Absolutely! You can use your own custom domain name with our platform. We provide instructions on how to set it up."},

    {"role": "user", "content": "Is there a limit to the number of pages I can create on my website?"},
    {"role": "assistant", "content": "The number of pages you can create depends on your chosen plan. Free plans may have limitations, while premium plans often offer unlimited pages. Check our pricing details for specifics."},

    {"role": "user", "content": "Do you offer website backup and restore options?"},
    {"role": "assistant", "content": "Yes, we provide regular website backups, and you can also manually create backups. In case of any issues, you can restore your website to a previous state."},

    {"role": "user", "content": "Can I use my website for blogging purposes?"},
    {"role": "assistant", "content": "Certainly! Our platform includes blogging features, making it easy to create and manage a blog on your website. You can publish articles, manage categories, and interact with your readers."},

    {"role": "user", "content": "Do you offer a free trial period for your premium plans?"},
    {"role": "assistant", "content": "Yes, we often provide a free trial period for our premium plans, allowing you to explore the full range of features before making a commitment. Check our website for current trial offers."},

    {"role": "user", "content": "Can I use my own website design and have you implement it?"},
    {"role": "assistant", "content": "Yes, you can provide your own website design, and our team can work with you to implement it on our platform. We offer design services to help bring your vision to life."},

    {"role": "user", "content": "What is the process for launching my website once it's created?"},
    {"role": "assistant", "content": "Launching your website is a straightforward process. We provide step-by-step instructions, including domain setup and publishing. Your website can be live in just a few clicks."},

    {"role": "user", "content": "Can I use your service to create a personal portfolio website?"},
    {"role": "assistant", "content": "Absolutely! Many users use our service to create personal portfolio websites to showcase their work, skills, and achievements. We have templates tailored for this purpose."},

    {"role": "user", "content": "Do you offer website analytics and reporting tools?"},
    {"role": "assistant", "content": "Yes, we provide built-in website analytics and reporting tools that allow you to track visitor data, traffic sources, and other important metrics to help you understand your website's performance."},

    {"role": "user", "content": "Can I add a contact form to my website for visitor inquiries?"},
    {"role": "assistant", "content": "Certainly! You can easily add a contact form to your website to allow visitors to get in touch with you. We offer customizable form options."},

    {"role": "user", "content": "What payment methods do you support for e-commerce transactions?"},
    {"role": "assistant", "content": "We support a wide range of payment methods, including credit cards, debit cards, PayPal, and other popular online payment options. You can configure your preferred payment methods in your e-commerce settings."},

    {"role": "user", "content": "Can I set up a blog on my business website to share industry updates?"},
    {"role": "assistant", "content": "Absolutely! Many businesses use blogs to share industry news, updates, and insights with their audience. You can easily integrate a blog into your business website."},

    {"role": "user", "content": "Do you offer multi-language support for websites targeting international audiences?"},
    {"role": "assistant", "content": "Yes, we support multi-language websites. You can create content in multiple languages and provide language switch options for your visitors. It's ideal for businesses with a global audience."},

    {"role": "user", "content": "What options do I have for website promotion and marketing on your platform?"},
    {"role": "assistant", "content": "We provide various marketing tools, including SEO optimization, email marketing integration, and social media sharing capabilities to help you promote your website effectively."},

    {"role": "user", "content": "Can I use my own images and media on my website, or are there restrictions on content?"},
    {"role": "assistant", "content": "You can use your own images and media on your website without restrictions. Our platform allows you to upload and customize content to suit your needs."},

    {"role": "user", "content": "Do you provide data migration services to move my content from another platform to yours?"},
    {"role": "assistant", "content": "Yes, we offer data migration services to help you transfer your content from another platform to ours. Our team can assist with the migration process."},

    {"role": "user", "content": "What level of technical expertise is required to use your website creation platform effectively?"},
    {"role": "assistant", "content": "Our platform is designed to be user-friendly, and no advanced technical expertise is required. However, having some basic familiarity with website design concepts can be helpful for customizing your site."},

    {"role": "user", "content": "Can I change my website's template after it's been created?"},
    {"role": "assistant", "content": "Yes, you can change your website's template even after it's been created. Our platform allows you to switch templates and retain your content and customization."},

    {"role": "user", "content": "Are there any restrictions on the types of content I can publish on my website?"},
    {"role": "assistant", "content": "While we have a policy against illegal or harmful content, we generally allow a wide range of content types. Please review our content guidelines on our website for more details."},

    {"role": "user", "content": "Can I sell digital products or downloads on my website using your e-commerce features?"},
    {"role": "assistant", "content": "Yes, our e-commerce features support the sale of digital products and downloads. You can list and sell digital items with ease."},

    {"role": "user", "content": "Do you offer templates specifically designed for business websites?"},
    {"role": "assistant", "content": "Yes, we have a selection of templates tailored for business websites. These templates often include features like contact forms, service listings, and more."},

    {"role": "user", "content": "Can I use my own CSS styles to customize my website's design further?"},
    {"role": "assistant", "content": "Yes, you can apply your own custom CSS styles to further customize your website's design. This is ideal for users with specific design preferences."},

    {"role": "user", "content": "Is it possible to add videos to my website, and if so, what hosting options are available?"},
    {"role": "assistant", "content": "Yes, you can add videos to your website. You can either upload videos directly to our platform, or you can embed videos from popular video hosting platforms like YouTube and Vimeo."},

    {"role": "user", "content": "Do you provide website backup and security options for protection against data loss and hacking?"}, 
    {"role": "assistant", "content": "Yes, we offer regular website backups and security features to protect against data loss and hacking attempts. Your website's security is a top priority for us."},

    {"role": "user", "content": "What is the process for adding a blog to my existing website?"},
    {"role": "assistant", "content": "Adding a blog to your existing website is simple. You can create a new page, choose a blog template, and start publishing articles. We provide easy-to-use blogging tools to get you started."},

    {"role": "user", "content": "Can I add a custom logo and branding to my website design?"},
    {"role": "assistant", "content": "Absolutely! You can upload your own custom logo and apply branding elements to your website design to create a cohesive and professional look."},

    {"role": "user", "content": "How do I optimize my website for search engines like Google?"}, 
    {"role": "assistant", "content": "We provide built-in SEO tools and guidance to help you optimize your website for search engines. This includes optimizing meta tags, titles, and content for better visibility."},

    {"role": "user", "content": "Is it possible to add a members-only section to my website for exclusive content?"}, 
    {"role": "assistant", "content": "Yes, you can create a members-only section on your website. You can restrict access to specific content or areas of your site to registered members."},

    {"role": "user", "content": "Can I connect my website to social media platforms for sharing content and updates?"}, 
    {"role": "assistant", "content": "Yes, you can easily connect your website to social media platforms. We offer social media sharing buttons and integration options to promote your content on various social networks."},

    {"role": "user", "content": "Do you offer website templates for creative portfolios, such as photography or art portfolios?"}, 
    {"role": "assistant", "content": "Yes, we have templates designed specifically for creative portfolios. These templates are ideal for showcasing photography, artwork, and other creative projects."},

    {"role": "user", "content": "What is the process for adding an online store to my existing website?"}, 
    {"role": "assistant", "content": "To add an online store, you can create a new section or page for your products, set up product listings, and configure payment options. Our platform provides step-by-step guidance for setting up an online store."},

    {"role": "user", "content": "Can I use my own domain for email addresses associated with my website?"}, 
    {"role": "assistant", "content": "Yes, you can use your own domain for email addresses associated with your website. We provide email hosting options and setup instructions."},

    {"role": "user", "content": "What is the process for adding a contact form to my website?"}, 
    {"role": "assistant", "content": "Adding a contact form is easy. You can select a contact form widget or element, customize its fields, and place it on the desired page. Visitors can use the form to send you inquiries."},

    {"role": "user", "content": "Is there a limit to the amount of storage space available for my website's content and media files?"}, 
    {"role": "assistant", "content": "The amount of storage space available may vary depending on your plan. Premium plans often offer larger storage capacities, while free plans may have limitations. Check our pricing details for specifics."},

    {"role": "user", "content": "Can I add a live chat feature to my website for customer support?"}, 
    {"role": "assistant", "content": "Yes, you can add a live chat feature to your website to provide real-time customer support. We offer integration with popular live chat tools."},

    {"role": "user", "content": "Do you offer SSL certificates for secure website connections?"}, 
    {"role": "assistant", "content": "Yes, we provide SSL certificates to ensure secure website connections. SSL encryption is essential for protecting user data and building trust with visitors."},

    {"role": "user", "content": "Can I schedule automated backups for my website?"}, 
    {"role": "assistant", "content": "Yes, you can schedule automated backups for your website to ensure regular data protection. You can configure backup settings to fit your needs."},

    {"role": "user", "content": "Can I track website traffic and visitor behavior using your platform?"}, 
    {"role": "assistant", "content": "Yes, our platform includes built-in analytics tools that allow you to track website traffic, visitor behavior, and other important metrics. You can gain insights into how your website performs."},

    {"role": "user", "content": "Do you offer a content management system (CMS) for managing website content?"}, 
    {"role": "assistant", "content": "Yes, our platform includes a content management system (CMS) that allows you to easily manage and update website content, including text, images, and media."},

    {"role": "user", "content": "Can I create password-protected pages or sections on my website for private content?"}, 
    {"role": "assistant", "content": "Yes, you can create password-protected pages or sections on your website to restrict access to authorized users. It's a useful feature for private content or member areas."}
  ]
}

# Fonction pour convertir le jeu de données au format GPT-3.5
def convert_to_gpt35_format(dataset):
    fine_tuning_data = []
    for message in dataset["messages"]:
        fine_tuning_data.append({
            "messages": [
                {"role": message["role"], "content": message["content"]}
            ]
        })
    return fine_tuning_data

def write_to_jsonl(data, file_path):
    with open(file_path, 'w') as file:
        for entry in data:
            json.dump(entry, file)
            file.write('\n')


# Diviser les données en ensembles d'entraînement et de validation
train_data, val_data = train_test_split(convert_to_gpt35_format(dataset), test_size=0.2, random_state=42)

# Écrire les ensembles d'entraînement et de validation au format JSONL
write_to_jsonl(train_data, "train.jsonl")
write_to_jsonl(val_data, "val.jsonl")

# Téléchargement des fichiers et démarrage du job de fine-tuning sur OpenAI
from openai import OpenAI

client = OpenAI(api_key="Bearer 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")
training_file = client.files.create(file=open("train.jsonl", "rb"), purpose="fine-tune")
validation_file = client.files.create(file=open("val.jsonl", "rb"), purpose="fine-tune")

response = client.fine_tuning.jobs.create(
    training_file=training_file.id,
    validation_file=validation_file.id,
    model="gpt-3.5-turbo",
    suffix="support_chatbot_fine_tuning"
)