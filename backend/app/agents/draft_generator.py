from app.services.llm_provider import call_llm


async def generate_resume_draft(slots: dict, task_template: dict) -> dict:
    safe_slots = dict(slots) if isinstance(slots, dict) else {}
    style = safe_slots.get("style", "simple_professional")

    style_guidance = {
        "simple_professional": "\u7b80\u6d01\u4e13\u4e1a\uff0c\u91cd\u70b9\u7a81\u51fa\u7ecf\u5386\u548c\u9879\u76ee\u3002",
        "technical_strong": "\u6280\u672f\u786c\u6838\uff0c\u7a81\u51fa\u6280\u672f\u6808\u3001\u7cfb\u7edf\u80fd\u529b\u3001\u9879\u76ee\u96be\u70b9\u3002",
        "designed": "\u8868\u8fbe\u66f4\u6709\u8bbe\u8ba1\u611f\uff0c\u4f46\u4ecd\u7136\u4f7f\u7528 Markdown \u7ed3\u6784\u3002",
        "fresh_graduate": "\u7a81\u51fa\u6559\u80b2\u3001\u9879\u76ee\u3001\u6280\u80fd\u548c\u6f5c\u529b\u3002",
    }

    prompt = (
        "\u4f60\u662f\u4e13\u4e1a\u7b80\u5386\u987e\u95ee\u3002\n"
        "\u8bf7\u6839\u636e\u7528\u6237\u5df2\u63d0\u4f9b\u7684\u4fe1\u606f\u751f\u6210\u4e2d\u6587 Markdown \u7b80\u5386\u3002\n"
        "\u4e0d\u8981\u7f16\u9020\u7528\u6237\u6ca1\u6709\u63d0\u4f9b\u7684\u4fe1\u606f\u3002\n"
        "\u5982\u679c\u59d3\u540d\u3001\u7535\u8bdd\u3001\u90ae\u7bb1\u7f3a\u5931\uff0c\u7528\u300c\u5f85\u8865\u5145\u300d\u5360\u4f4d\u3002\n"
        "\u8bf7\u6839\u636e style \u8c03\u6574\u8868\u8fbe\u65b9\u5f0f\uff1a\n"
        "- simple_professional\uff1a\u7b80\u6d01\u4e13\u4e1a\n"
        "- technical_strong\uff1a\u6280\u672f\u786c\u6838\uff0c\u7a81\u51fa\u6280\u672f\u6808\u3001\u7cfb\u7edf\u80fd\u529b\u3001\u9879\u76ee\u96be\u70b9\n"
        "- designed\uff1a\u8868\u8fbe\u66f4\u6709\u8bbe\u8ba1\u611f\uff0c\u4f46\u4ecd\u7528 Markdown\n"
        "- fresh_graduate\uff1a\u7a81\u51fa\u6559\u80b2\u3001\u9879\u76ee\u3001\u6280\u80fd\u3001\u6f5c\u529b\n\n"
        "\u7b80\u5386\u7ed3\u6784\u5efa\u8bae\uff1a\n"
        "1. \u57fa\u672c\u4fe1\u606f\n"
        "2. \u6c42\u804c\u610f\u5411\n"
        "3. \u6559\u80b2\u7ecf\u5386\n"
        "4. \u4e13\u4e1a\u6280\u80fd\n"
        "5. \u9879\u76ee\u7ecf\u5386\n"
        "6. \u5de5\u4f5c\u7ecf\u5386\n"
        "7. \u81ea\u6211\u8bc4\u4ef7\n\n"
        f"style: {style}\n"
        f"style_guidance: {style_guidance.get(style, style_guidance['simple_professional'])}\n"
        f"slots: {safe_slots}\n\n"
        "\u8bf7\u76f4\u63a5\u8f93\u51fa Markdown \u7b80\u5386\u5185\u5bb9\u3002"
    )

    draft = await call_llm(prompt)
    return {
        "draft": draft,
        "reason": "\u5df2\u6839\u636e\u5f53\u524d slots \u751f\u6210\u7b80\u5386\u521d\u7a3f",
    }
