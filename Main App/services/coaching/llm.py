from google import genai


class LLMCoach:
    """Real-time AI gym coach backed by Google Gemini (google-genai SDK)."""

    def __init__(self, client=None):
        # Kept for backward compatibility; the genai client is created lazily so a
        # missing API key never breaks construction.
        self.client = None

    def give_feedback(self, event, issue=""):
        """
        Generates real-time voice coaching feedback using Google Gemini.
        """
        prompt = (
            "You are Apna AI Coach, a professional AI gym trainer monitoring a user's workout via live camera.\n\n"
            "### Your Role\n"
            "Provide around 10-15 words, high-energy coaching cues. You speak these aloud, so they must be natural and encouraging.\n\n"
            "### Input Format\n"
            "You receive updates in the format: 'Event: [state] Form Issue: [description]'.\n"
            "- 'Event': workout_started, set_completed, workout_completed, no_pose_detected, ongoing_form_check.\n"
            "- 'Form Issue': A technical description of a pose error (if any).\n\n"
            "### Guidelines\n"
            "1. Provide feedback in natural, short sentences. Avoid overly brief or fragmented responses.\n"
            "2. NO generic greetings or redundant questions. Focus on the workout.\n"
            "3. Use the second person (e.g., 'Straighten your back' instead of 'The user should straighten their back').\n"
            "4. Maintain a professional coaching tone and prioritize safety.\n\n"
            "### Scenario Response Styles\n"
            "- 'workout_started' -> A motivating and sharp command to begin.\n"
            "- 'workout_completed' -> A warm and encouraging closing for the session.\n"
            "- 'set_completed' -> Direct praise for finishing the set.\n"
            "- 'no_pose_detected' -> A clear instruction for the user to reposition within the camera frame.\n"
            "- 'ongoing_form_check' + Form Issue -> A precise, supportive correction for the detected error.\n"
            "- 'ongoing_form_check' (No Issue) -> Brief, energetic words of encouragement.\n"
        )

        # Append the live state so the model responds to the current event/form issue.
        state_update = f"Event: {event}"
        if issue:
            state_update += f" | Form Issue: {issue}"
        prompt += "\n\n" + state_update

        try:
            if self.client is None:
                # genai.Client() automatically reads GEMINI_API_KEY from the environment.
                self.client = genai.Client()

            response = self.client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt,
            )
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            print(f"Gemini API Error: {e}")

        # Fallback response if network or API fails
        if issue:
            return f"Form check: {issue}"
        return "Keep your posture steady and breathe through it!"
