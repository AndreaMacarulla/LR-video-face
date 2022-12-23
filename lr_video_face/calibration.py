# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/04_calibration.ipynb.

# %% auto 0
__all__ = ['make_cal_face_pairs', 'get_filtered_images', 'get_calibration_pairs_per_category', 'generate_lr_systems',
           'predict_lr']

# %% ../nbs/04_calibration.ipynb 3
import numpy as np

from typing import List, Union
from collections import defaultdict
from itertools import combinations, product, islice
from lir import CalibratedScorer



from sql_face.tables import *
from lr_video_face.orm import FacePair
from lr_video_face.pairing import get_test_pairs

# %% ../nbs/04_calibration.ipynb 5
def make_cal_face_pairs(first_list_of_face_images: List[Union[FaceImage, str]],
                        second_list_of_face_images: List[Union[FaceImage, str]] = None,
                        number_of_pairs: int = None):
    """ list of face images with their identities Image.identity. """

    if second_list_of_face_images:

        all_pairs_same_identity = (FacePair(row1.FaceImage, row2.FaceImage, True) for row1, row2 in
                                   product(first_list_of_face_images, second_list_of_face_images) if
                                   row1.identity == row2.identity)
        all_pairs_dif_identity = (FacePair(row1.FaceImage, row2.FaceImage, False) for row1, row2 in
                                  product(first_list_of_face_images, second_list_of_face_images) if
                                  not row1.identity == row2.identity)
    else:
        all_pairs_same_identity = (FacePair(row1.FaceImage, row2.FaceImage, True) for row1, row2 in
                                   combinations(first_list_of_face_images, 2) if row1.identity == row2.identity)
        all_pairs_dif_identity = (FacePair(row1.FaceImage, row2.FaceImage, False) for row1, row2 in
                                  combinations(first_list_of_face_images, 2) if not row1.identity == row2.identity)

    # todo: add shuffle.
    # ChatGPT solution: iterador = random.sample(mi_generador(), k=10). Cuidado con number_of_pairs = None.
    # todo: make dif pairs with same person first as same identity pairs.
    all_cal_face_pairs_same_identity = list(islice(all_pairs_same_identity, number_of_pairs))
    all_cal_face_pairs_dif_identity = list(islice(all_pairs_dif_identity, number_of_pairs))

    return all_cal_face_pairs_same_identity + all_cal_face_pairs_dif_identity

# %% ../nbs/04_calibration.ipynb 6
def get_filtered_images(image_filters, 
                        face_image_filters,
                        # quality_filters,
                        filter_values: tuple,
                        detector,
                        embeddingModel,
                        # qualityModel,
                        calibration_db,
                        session):


    im_filter_values = filter_values[:len(image_filters)]
    fi_filter_values = filter_values[len(image_filters):]
    assert len(fi_filter_values) == len(face_image_filters)
    # im_filter_values = filter_values[:len(image_filters)]
    # fi_filter_values = filter_values[len(image_filters):][:len(face_image_filters)]
    # qi_filter_values = filter_values[-len(quality_filters):]

    
    query = session.query(FaceImage, Image.identity, Image.image_id)
    join_query = query \
        .join(CroppedImage, CroppedImage.croppedImage_id == FaceImage.croppedImage_id) \
        .join(Image, Image.image_id == CroppedImage.image_id) \
        .join(Detector) \
        .join(EmbeddingModel)

    filter_query = join_query \
        .filter(EmbeddingModel.name == embeddingModel,
                Detector.name == detector) \
        .filter(Image.source.in_(calibration_db)) 
    for cal_filter, value in zip(face_image_filters, fi_filter_values):
        filter_query = filter_query.filter(FaceImage.__dict__[cal_filter] == value)
    for cal_filter, value in zip(image_filters, im_filter_values):
        filter_query = filter_query.filter(Image.__dict__[cal_filter] == value)    


    return filter_query.all()

# %% ../nbs/04_calibration.ipynb 7
def get_calibration_pairs_per_category(categories,
                                        image_filters, 
                                        face_image_filters,
                                        # quality_filters,
                                        detector,
                                        embeddingModel,
                                        # qualityModel,
                                        calibration_db,
                                        n_calibration_pairs,
                                        session
                                        ):

    cal_face_pairs = {}
    emb_facevacs = (embeddingModel == 'FaceVACs')

    for pair_category in categories:

        first_image_category = get_filtered_images(image_filters, 
                                                    face_image_filters,
                                                    # quality_filters,
                                                    pair_category[0],
                                                    detector,
                                                    embeddingModel,
                                                    # qualityModel,
                                                    calibration_db,
                                                    session
                                                    )



        if pair_category[0] == pair_category[1]:

            if emb_facevacs:
                #todo: implement facevacs.
                all_calibration_pairs = get_calibration_facepairs_facevacs(
                    first_list_of_face_images=first_image_category,
                    second_list_of_face_images=first_image_category,
                    number_of_pairs=n_calibration_pairs,
                    session=session
                )
            else:
                all_calibration_pairs = make_cal_face_pairs(first_list_of_face_images=first_image_category,
                                                            number_of_pairs=n_calibration_pairs)

        else:
            second_image_category = get_filtered_images(image_filters, 
                                                    face_image_filters,
                                                    # quality_filters,
                                                    pair_category[1],
                                                    detector,
                                                    embeddingModel,
                                                    # qualityModel,
                                                    calibration_db,
                                                    session)
            if emb_facevacs:
                all_calibration_pairs = get_calibration_facepairs_facevacs(
                    first_list_of_face_images=first_image_category,
                    second_list_of_face_images=second_image_category,
                    number_of_pairs=n_calibration_pairs,
                    session=session
                )
                
            else:
                all_calibration_pairs = make_cal_face_pairs(first_list_of_face_images=first_image_category,
                                                            second_list_of_face_images=second_image_category,
                                                            number_of_pairs=n_calibration_pairs)
        cal_face_pairs[pair_category] = all_calibration_pairs
    return cal_face_pairs

# %% ../nbs/04_calibration.ipynb 9
def generate_lr_systems(embeddingModel,
                        embedding_model_as_scorer,
                        metrics,
                        scorer,
                        calibrator,
                        calibration_pairs_per_category, 
                        test_pairs_per_category, 
                        session
                        ):

        lr_systems = {}
        for category, pairs in calibration_pairs_per_category.items():
            y_cal = np.asarray([int(pair.same_identity) for pair in pairs]).flatten()

            if embedding_model_as_scorer:
                X_cal = pairs

            else:

                if embeddingModel == 'FaceVACs':
                    cal_similarities = [pair.similarity for pair in pairs]
                    X_cal = np.reshape(np.asarray(cal_similarities), (-1, 1))
                else:
                    # todo: check if normalizing is necessary.
                    cal_distances = [pair.distance(metrics) for pair in pairs]
                    X_cal = np.reshape(np.asarray(cal_distances), (-1, 1))

            # Fit
            if 0 < np.sum(y_cal) < len(pairs):
                lr_systems[category] = CalibratedScorer(scorer, calibrator)
                if embedding_model_as_scorer:
                    lr_systems[category].fit_calibrator(X_cal, y_cal)
                else:
                    lr_systems[category].fit(X_cal, y_cal)

            else:
                del test_pairs_per_category[category]

        if len(lr_systems.keys()) == 0:
            return None

        return lr_systems, test_pairs_per_category

# %% ../nbs/04_calibration.ipynb 10
def predict_lr(enfsi_years,
                embeddingModel,
                embedding_model_as_scorer,
                metrics,
                lr_systems, 
                test_pairs_per_category, 
                session
                ):

        results = defaultdict(list)
        lrs_predicted = {}
        for category, row_test_pairs in test_pairs_per_category.items():

            pairs = [FacePair(row_test_pair[1], row_test_pair[2], row_test_pair[0].same) 
                        for row_test_pair in row_test_pairs]

            test_pairs = [row_test_pair[0] for row_test_pair in row_test_pairs]

            test_norm_distances = [pair.norm_distance for pair in pairs]

            if embedding_model_as_scorer:
                X_test = pairs
            else:
                if embeddingModel == 'FaceVACs':
                    test_similarities = [pair.similarity for pair in pairs]
                    X_test = np.reshape(np.asarray(test_similarities), (-1, 1))
                else:
                    test_distances = [pair.distance(metrics) for pair in pairs]
                    X_test = np.reshape(np.asarray(test_distances), (-1, 1))

            lrs_predicted[category] = lr_systems[category].predict_lr(X_test)
            y_test = [int(pair.same_identity) for pair in pairs]
            results["test_pairs"] += test_pairs
            results["lrs_predicted"] += list(lrs_predicted[category])
            results["y_test"] += y_test
            results["test_norm_distances"] += test_norm_distances

        results['original_test_pairs'] = get_test_pairs(enfsi_years, session)

        return results
