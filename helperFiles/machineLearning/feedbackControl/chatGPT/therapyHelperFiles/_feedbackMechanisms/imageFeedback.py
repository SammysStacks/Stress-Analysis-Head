# General
from datetime import date
import itertools
import time
import os

# Import Files for Machine Learning
from ..imageModifications import imageModifications  # Methods for working with and altering images.
from ..browserControl import browserControl      # Methods for controlling the web browser.


class ImageFeedback:

    image_counter = itertools.count()

    def __init__(self, client, model, thread, txtFilePath, userName="Subject_XYZ"):
        # General parameters.
        self.imageGenerationEvent = thread
        self.txtFilePath = txtFilePath
        self.userName = userName
        self.imageModel = model
        self.client = client

        # Instantiate the necessary classes.
        self.imageController = imageModifications(os.path.dirname(__file__) + "/_savedImages/")
        self.browserController = browserControl()

    # ---------------------------- Image Methods ---------------------------- #
        
    def imageThread(self, conversationHistory, resultContainer):
        textPrompt, textPromptForImage = self.prepPromptForImage(conversationHistory)
        response = self.getImageResponse(textPromptForImage)

        self.saveImage(response, textPromptForImage)
        self.displayImage(response)

        resultContainer['response'] = response
        return response

    @staticmethod
    def prepPromptForImage(conversationHistory):
        textPrompt = conversationHistory[-1]['content'][0]['text']
        i = -2
        mostRecentAssistantContent = []
        while len(mostRecentAssistantContent) < 2:
            if conversationHistory[i]['role'] != 'user':
                mostRecentAssistantContent.append(conversationHistory[i])
            i -= 1
        if textPrompt != '':
            textPromptForImage = f"Previously, you said: {mostRecentAssistantContent[0]['content'][0]['text']} and {mostRecentAssistantContent[1]['content'][0]['text']}. Now, \
                the user said: {textPrompt}. As a friendly and helpful virtual therapist, please think about the subject and generate a relevant image to help them feel better."
            return textPrompt, textPromptForImage
        else:
            textPromptForImage = f"Previously, you said: {mostRecentAssistantContent[0]['content'][0]['text']} and {mostRecentAssistantContent[1]['content'][0]['text']}. \
                As a friendly and helpful virtual therapist, please think about the subject's experience so far and generate a relevant, calming, and different image to help them feel better and place them in a new world. Add no words to the image."
            return mostRecentAssistantContent[1]['content'][0]['text'][:20], textPromptForImage

    def getImageResponse(self, textPrompt):
        assert len(textPrompt) <= 4000, f"The maximum length is 4000 characters for text. Given {len(textPrompt)} characters"
        assert isinstance(textPrompt, str), f"Expecting the text prompt to be a string. Given type {type(textPrompt)}. Value: {textPrompt}"

        self.imageGenerationEvent.clear()

        response = self.client.images.generate(
            model=self.imageModel,
            user=self.userName,
            prompt=textPrompt,
            size="1024x1024",
            style="vivid",
            quality="hd",
            n=1,
        )

        return response

    def displayImage(self, response):
        # Get the image URL.
        image_url = self.getImageURL(response)
        # Open the image URL with the webdriver.
        self.browserController.open_url(image_url)

    def saveImage(self, response, org_prompt, save_path="/therapyHelperFiles/_savedImages/"):  # change this to therapyHelperFiles/_savedImages/ for the right directory name
        # Get the image from image URL.
        image_url = self.getImageURL(response)
        imageRGBA = self.imageController.pullDownWebImage(image_url)

        # Make an image path
        filepath = os.path.dirname(__file__) + "/../../" + save_path
        if not os.path.exists(filepath):
            print('path does not exist', filepath)

        # Save the file with today's date
        counter = next(self.image_counter)
        image_filepath = os.path.join(filepath, f"{date.today()}_{1}.png")
        imageRGBA.save(image_filepath, 'PNG')
        # print(f"Image saved to {image_filepath}")

        # Write this to conversation history txt file
        file = open(self.txtFilePath, 'a')
        file.write(f"*** Image saved to {image_filepath} *** \n")
        file.close()

        return image_filepath
        
    @staticmethod
    def getImageURL(response):
        return response.data[0].url
