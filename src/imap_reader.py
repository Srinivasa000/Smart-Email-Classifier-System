import imaplib
import email
from email.header import decode_header


class IMAPReader:

    def __init__(self, username, password, server, port=993):
        self.username = username
        self.password = password
        self.server = server
        self.port = port

    def connect(self):
        try:
            mail = imaplib.IMAP4_SSL(self.server, self.port)
            mail.login(self.username, self.password)
            return mail
        except Exception as e:
            raise Exception(f"IMAP login failed: {e}")

    def fetch_recent(self, limit=20):
        """
        Fetch the most recent emails (default: last 20)
        Returns a list of dicts with subject, body, from, id
        """
        mail = self.connect()
        mail.select("INBOX")

        _, msg_ids = mail.search(None, "ALL")
        msg_list = msg_ids[0].split()

        if not msg_list:
            return []

        emails = []

        for msg_id in msg_list[-limit:]:
            _, msg_data = mail.fetch(msg_id, "(RFC822)")
            raw_msg = msg_data[0][1]
            msg = email.message_from_bytes(raw_msg)

            # Decode subject
            raw_subject = msg.get("Subject", "")
            subject, encoding = decode_header(raw_subject)[0]
            if isinstance(subject, bytes):
                subject = subject.decode(encoding or "utf-8", errors="ignore")

            sender = msg.get("From", "")

            # Decode body safely
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        try:
                            body = part.get_payload(decode=True).decode(errors="ignore")
                        except:
                            body = ""
                        break
            else:
                try:
                    body = msg.get_payload(decode=True).decode(errors="ignore")
                except:
                    body = ""

            emails.append({
                "subject": subject,
                "body": body,
                "from": sender,
                "id": msg_id
            })

        return list(reversed(emails))

    def move_to_folder(self, msg_id, folder):
        """
        Move the email to another folder.
        Only works if server supports folder operations.
        """
        mail = self.connect()
        mail.select("INBOX")
        try:
            mail.copy(msg_id, folder)
            mail.store(msg_id, "+FLAGS", "\\Deleted")
            mail.expunge()
        except Exception as e:
            raise Exception(f"Failed to move email: {e}")
