from tortoise.models import Model
from tortoise import fields
from enum import Enum

class Role(str, Enum):
    ADMIN = "ADMIN"
    AGENT = "AGENT"

class NoteStatus(str, Enum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    DONE = "DONE"
    FAILED = "FAILED"

class User(Model):
    id = fields.IntField(pk=True)
    email = fields.CharField(max_length=255, unique=True)
    password = fields.CharField(max_length=255)
    role = fields.CharEnumField(Role, default=Role.AGENT)
    created_at = fields.DatetimeField(auto_now_add=True)
    
    # Reverse relation to notes
    notes: fields.ReverseRelation["Note"]

    class Meta:
        table = "users"

class Note(Model):
    id = fields.IntField(pk=True)
    raw_text = fields.TextField()
    summary = fields.TextField(null=True)
    status = fields.CharEnumField(NoteStatus, default=NoteStatus.QUEUED)
    user = fields.ForeignKeyField("models.User", related_name="notes")
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "notes"
