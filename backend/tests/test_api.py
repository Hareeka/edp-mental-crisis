def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_auth_flow(client):
    r = client.post("/api/auth/register", json={"alias": "a", "email": "flow@ex.com", "password": "password123"})
    assert r.status_code == 201
    assert client.post("/api/auth/register", json={"alias": "a", "email": "flow@ex.com",
                                                   "password": "password123"}).status_code == 409
    bad = client.post("/api/auth/token", data={"username": "flow@ex.com", "password": "nope"})
    assert bad.status_code == 401
    tok = client.post("/api/auth/token", data={"username": "flow@ex.com", "password": "password123"}).json()
    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {tok['access_token']}"})
    assert me.json()["email"] == "flow@ex.com"


def test_requires_auth(client):
    assert client.post("/api/chat", json={"message": "hi"}).status_code == 401
    assert client.get("/api/moods").status_code == 401


def test_chat_low_risk(client, auth):
    r = client.post("/api/chat", json={"message": "Had a nice walk, feeling pretty good today"}, headers=auth)
    assert r.status_code == 200
    body = r.json()
    assert body["risk"]["level"] == "low"
    assert body["sentiment"] in {"positive", "neutral", "negative"}
    assert "not a diagnosis" in body["disclaimer"]


def test_chat_high_risk_returns_safety(client, auth):
    r = client.post("/api/chat", json={"message": "I want to end my life, I have the pills ready"}, headers=auth)
    body = r.json()
    assert body["risk"]["level"] == "high"
    assert body["reply_mode"] == "safety"
    assert body["resources"]


def test_chat_validation(client, auth):
    assert client.post("/api/chat", json={"message": "   "}, headers=auth).status_code == 422
    assert client.post("/api/chat", json={"message": "x" * 6000}, headers=auth).status_code == 422
    assert client.post("/api/chat", json={"message": "hi", "mood": "bogus"}, headers=auth).status_code == 422


def test_messages_encrypted_at_rest(client, auth):
    from sqlalchemy import select

    from app.db import SessionLocal
    from app.models import Interaction

    secret = "my unique secret sentence 12345"
    iid = client.post("/api/chat", json={"message": secret}, headers=auth).json()["interaction_id"]
    with SessionLocal() as db:
        stored = db.scalar(select(Interaction.message).where(Interaction.id == iid))
    assert secret not in stored
    hist = client.get("/api/interactions", headers=auth).json()
    assert any(h["message"] == secret for h in hist)


def test_moods_trends_dashboard(client, auth):
    for mood, i in [("low", 6), ("okay", 4), ("good", 3)]:
        assert client.post("/api/moods", json={"mood": mood, "intensity": i, "note": "n"}, headers=auth).status_code == 201
    assert client.post("/api/moods", json={"mood": "good", "intensity": 11}, headers=auth).status_code == 422
    assert len(client.get("/api/moods", headers=auth).json()) == 3
    t = client.get("/api/trends?days=7", headers=auth).json()
    assert len(t) == 7 and t[-1]["mood_count"] == 3 and t[-1]["avg_mood"] == 3.0
    d = client.get("/api/dashboard", headers=auth).json()
    assert d["mood_average_7d"] == 3.0


def test_feedback(client, auth):
    iid = client.post("/api/chat", json={"message": "hello"}, headers=auth).json()["interaction_id"]
    r = client.post("/api/feedback", json={"interaction_id": iid, "usefulness": 4, "relevance": 5}, headers=auth)
    assert r.status_code == 201


def test_user_isolation(client, auth):
    import uuid

    other = client.post("/api/auth/register", json={"alias": "o", "email": f"{uuid.uuid4().hex}@ex.com",
                                                    "password": "password123"}).json()["access_token"]
    client.post("/api/chat", json={"message": "private words"}, headers=auth)
    hist = client.get("/api/interactions", headers={"Authorization": f"Bearer {other}"}).json()
    assert hist == []


def test_admin_requires_admin(client, auth):
    assert client.get("/api/admin/stats", headers=auth).status_code == 403


def test_delete_account(client):
    tok = client.post("/api/auth/register", json={"alias": "d", "email": "del@ex.com",
                                                  "password": "password123"}).json()["access_token"]
    h = {"Authorization": f"Bearer {tok}"}
    client.post("/api/chat", json={"message": "hello"}, headers=h)
    assert client.delete("/api/auth/me", headers=h).status_code == 204
    assert client.get("/api/auth/me", headers=h).status_code == 401
