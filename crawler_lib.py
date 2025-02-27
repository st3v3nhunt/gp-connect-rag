from typing import List

def get_urls_to_crawl() -> List[str]:
    """Get the list of URLs to crawl."""
    # TODO: Generate list of URLs to crawl, possibly using the sitemap(s)
    return [
        # not included as not useful
        # "https://developer.nhs.uk/apis/gpconnect-1-6-0/system_demonstrator.html",
        # "https://developer.nhs.uk/apis/gpconnect-1-6-0/system_swagger.html",
        # "https://developer.nhs.uk/apis/gpconnect-1-6-0/system_reference_postman.html",
        # not included due to context length being too large i.e. over 8192 tokens
        # "https://developer.nhs.uk/apis/gpconnect-1-6-0/development_fhir_api_guidance.html", # context of page is 10486 tokens, max is 8192!
        # "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_fhir_examples_pathology.html", # context of page is 10947 tokens, max is 8192!
        # "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_retrieve_patient_record.html", # context of page is 14791 tokens, max is 8192!
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/index.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/overview_engage.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/overview_priority_capabilities.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/designprinciples_open_api_principles.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/designprinciples_development_principles.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/designprinciples_data_model_principles.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/designprinciples_ig_principles.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/designprinciples_clinical_safety_principles.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/designprinciples_assurance_principles.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/design_clinical_terminologies.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/design_product_versioning.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/development_fhir_open_source_guidance.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/development_general_api_guidance.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/development_branch_surgeries.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/development_fhir_resource_guidance.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/development_api_security_guidance.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/development_fhir_error_handling_guidance.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/development_api_volume_and_performance.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/development_api_non_functional_requirements.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/foundations.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/appointments.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/access_documents.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/send_document.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/integration_illustrated.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/integration_system_topologies.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/integration_cross_organisation_audit_and_provenance.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/integration_personal_demographic_service.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/integration_spine_directory_service.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/integration_sds_registering_endpoints.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/integration_interaction_ids.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/integration_spine_secure_proxy.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/testing_deliverables.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/testing_environments.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/testing_api_provider_testing.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/testing_api_consumer_testing.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/support_faq.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/overview_glossary.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/support_communications.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_requirements.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_known_issues.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_resources_overview.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_lists_for_message_structure.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_linkages.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_search.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_searchMultiAreaSearches.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_searchExamples.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_list.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_bundle.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_allergies_guidance.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_allergyintolerance.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_fhir_examples_allergies.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_medication_resource_relationships.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_medication_guidance.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_medication.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_medicationstatement.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_medicationrequest.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_fhir_examples_medication.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_immunization_guidance.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_immunization.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_fhir_examples_immunizations.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_uncategorisedData_guidance.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_observation_uncategorisedData.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_observation_bloodPressure.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_fhir_examples_uncategorised.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_consultation_guidance.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_encounter.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_list_consultation.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_fhir_examples_consultations.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_problems_guidance.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_problems.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_fhir_examples_consultations.html#",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_pathology_guidance.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_DiagnosticReport.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_specimen.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_observation_testGroup.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_observation_testResult.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_observation_filingComments.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_ProcedureRequest.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_referralrequest_guidance.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_referralrequest.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_fhir_examples_referrals.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_documents_guidance.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_diaryentry_guidance.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_diaryentry.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_fhir_examples_diaryentries.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_migrate_patient_record.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_version_compatibility.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_clinical_area_config.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_development_fhir_examples_forwards_consultations.html",
        "https://developer.nhs.uk/apis/gpconnect-1-6-0/accessrecord_structured_get_the_fhir_capability_statement.html",
    ]
