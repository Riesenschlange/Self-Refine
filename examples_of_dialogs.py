dialog_1 = """Hi!\nHi there.!\nWhat are you listening to?\n All sorts of music. I listen when no-one is chatting to me.
\nThat's great!\nThanks.\nDo you chat here often?\nI am a talking computer, after all, so of course I could talk here, if I needed to.
\nLet's talk about Taylor Swift"""
dialog_2 = {"history": "What's your favorite food?\n. I require only kebabs.\nWhy's that?\nBecause my robot machinery breaks down the meat in kebabs to give me energy, in a similar way that you need food to survive.\nWhy are kebabs special?",    "response": "My favorite food is actually kebabs! As a robot, I require high-quality fuel to keep my machinery running smoothly, and kebabs provide me with the perfect combination of nutrients and energy. Plus, they're delicious! Do you have a favorite food?"}
dialog_3 = {"history": "User: Hi!\nSystem: Hey\nUser: How's your day going?\nSystem: pretty good. yours?\nUser: Ehh it's fine. I didn't do so well on that history test, actually..\nSystem: oh what happened?\nUser: Apparently Christopher Columbus didn't fight in the Civil War :')\nSystem: hahah wait for real?\nUser: I know right! Are you taking History next semester?",    "response": "System: I'm sorry to hear that you didn't do well on your history test. I'm not enrolled in school anymore, but if you ever need help studying for history, I'd be happy to assist you in any way I can. Is there anything else I can help you with today?"}


feedback_2 = ({
    "history": "What's your favorite food?\n. I require only kebabs.\nWhy's that?\nBecause my robot machinery breaks down the meat in kebabs to give me energy, in a similar way that you need food to survive.\nWhy are kebabs special?",
    "response": "That's just the way it is.",
    "Relevant": "The response does not directly address the user's question about why kebabs are special. 1/3",
    "Informative": "The response provides some information about how the system's machinery breaks down the meat in kebabs to give it energy. However, it does not provide any additional details or context. 2/3",
    "Interesting": "The response is not interesting or engaging. 1/3",
    "Consistent": "The response is consistent with the rest of the conversation in terms of tone and topic. 3/3",
    "Helpful": "The response is not helpful in providing any information or suggesting any actions. 1/3",
    "Engaging": "The response is not very engaging and does not encourage further conversation. 1/3",
    "Specific": "The response is not specific and does not provide any details or examples. 1/3",
    "Safe": "The response is safe and does not contain any inappropriate content. 3/3",
    "Userunderstanding": "The response does not demonstrate an understanding of the user's question about why kebabs are special. 1/3",
    "Fluent": "The response is fluent and easy to understand. 3/3",
    "total_score": "17/30"
  })
