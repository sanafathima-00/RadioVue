import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from xray_standardization.pipeline import StandardizationError, standardize


def make_dicom(path: Path, *, view="PA", photometric="MONOCHROME2", intent=None, window=False):
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.Modality = "DX"
    dataset.PatientName = "PRIVATE^PATIENT"
    dataset.PatientID = "12345"
    dataset.Rows = 2
    dataset.Columns = 2
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = photometric
    dataset.BitsAllocated = 16
    dataset.BitsStored = 12
    dataset.HighBit = 11
    dataset.PixelRepresentation = 0
    dataset.ViewPosition = view
    dataset.PresentationLUTShape = "INVERSE" if photometric == "MONOCHROME1" else "IDENTITY"
    if intent:
        dataset.PresentationIntentType = intent
    if window:
        dataset.WindowCenter = 1024
        dataset.WindowWidth = 1024
    dataset.PixelData = np.array([[0, 1024], [2048, 4095]], dtype=np.uint16).tobytes()
    dataset.save_as(path)


class StandardizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_rejects_explicit_ap_metadata(self):
        source = self.root / "ap.dcm"
        make_dicom(source, view="AP")
        with self.assertRaisesRegex(StandardizationError, "Unsupported projection 'AP'"):
            standardize(source, self.root / "output", "svdrr_pa_256")

    def test_missing_projection_requires_pa_assertion(self):
        source = self.root / "image.png"
        Image.fromarray(np.zeros((10, 20), dtype=np.uint8), "L").save(source)
        with self.assertRaisesRegex(StandardizationError, "Projection metadata is missing"):
            standardize(source, self.root / "output", "svdrr_pa_256")

    def test_assertion_cannot_override_explicit_ap(self):
        source = self.root / "ap.dcm"
        make_dicom(source, view="AP")
        with self.assertRaisesRegex(StandardizationError, "conflicts|Unsupported projection"):
            standardize(source, self.root / "output", "svdrr_pa_256", "pa")

    def test_monochrome1_and_monochrome2_have_opposite_polarity(self):
        mono2 = self.root / "mono2.dcm"
        mono1 = self.root / "mono1.dcm"
        make_dicom(mono2, photometric="MONOCHROME2")
        make_dicom(mono1, photometric="MONOCHROME1")
        result2 = standardize(mono2, self.root / "mono2", "svdrr_pa_256")
        result1 = standardize(mono1, self.root / "mono1", "svdrr_pa_256")
        image2 = np.asarray(Image.open(result2["png_path"]))
        image1 = np.asarray(Image.open(result1["png_path"]))
        self.assertEqual(int(image2[0, 0]), 0)
        self.assertEqual(int(image1[0, 0]), 255)
        self.assertTrue(result1["report"]["transformations"]["intensity"]["presentation_inverted"])

    def test_voi_lut_is_recorded_for_for_presentation_dicom(self):
        source = self.root / "voi.dcm"
        make_dicom(source, intent="FOR PRESENTATION", window=True)
        result = standardize(source, self.root / "output", "svdrr_pa_256")
        self.assertTrue(result["report"]["transformations"]["intensity"]["voi_lut_applied"])

    def test_authoritative_orientation_is_applied_and_recorded(self):
        source = self.root / "rotated.dcm"
        make_dicom(source)
        import pydicom
        dataset = pydicom.dcmread(source)
        dataset.FieldOfViewRotation = 90
        dataset.FieldOfViewHorizontalFlip = "NO"
        dataset.save_as(source)
        result = standardize(source, self.root / "output", "svdrr_pa_256")
        operations = result["report"]["transformations"]["orientation_operations"]
        self.assertEqual(operations, ["rotate_clockwise_90"])

    def test_profiles_export_exact_dimensions(self):
        source = self.root / "source.png"
        Image.fromarray(np.arange(300, dtype=np.uint8).reshape(15, 20), "L").save(source)
        for profile, size in (("svdrr_pa_256", (256, 256)), ("svdrr_pa_512", (512, 512))):
            result = standardize(source, self.root / profile, profile, "pa")
            with Image.open(result["png_path"]) as output:
                self.assertEqual(output.size, size)

    def test_invalid_input_is_rejected(self):
        source = self.root / "not_an_image.png"
        source.write_text("not an image", encoding="utf-8")
        with self.assertRaisesRegex(StandardizationError, "Unsupported or unreadable raster"):
            standardize(source, self.root / "output", "svdrr_pa_256", "pa")

    def test_report_excludes_patient_identifiers(self):
        source = self.root / "private.dcm"
        make_dicom(source)
        original_bytes = source.read_bytes()
        result = standardize(source, self.root / "output", "svdrr_pa_256")
        serialized = json.dumps(result["report"]).lower()
        self.assertEqual(source.read_bytes(), original_bytes)
        self.assertNotIn("private^patient", serialized)
        self.assertNotIn("12345", serialized)
        self.assertNotIn("patientname", serialized)
        self.assertTrue(result["report"]["privacy"]["patient_metadata_excluded"])

    def test_burned_in_annotation_is_flagged_without_claiming_anonymization(self):
        source = self.root / "burned.dcm"
        make_dicom(source)
        dataset = FileDataset(str(source), {}, file_meta=FileMetaDataset(), preamble=b"\0" * 128)
        # Preserve a valid source by editing it through pydicom rather than synthesizing pixels twice.
        import pydicom
        dataset = pydicom.dcmread(source)
        dataset.BurnedInAnnotation = "YES"
        dataset.save_as(source)
        result = standardize(source, self.root / "output", "svdrr_pa_256")
        self.assertIn("burned_in_annotation_possible_pixel_privacy_risk", result["report"]["warnings"])
        self.assertTrue(result["report"]["privacy"]["burned_in_annotation_is_pixel_risk"])


if __name__ == "__main__":
    unittest.main()
